import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy import linalg
import os
import shutil
import argparse
import time

# CVXOPT kontrolü
try:
    import cvxopt
    import cvxopt.solvers
    cvxopt.solvers.options['show_progress'] = False
    CVXOPT_AVAILABLE = True
except ImportError:
    print("Uyarı: cvxopt yüklü değil. Sadece SciPy çözücü kullanılabilir.")
    CVXOPT_AVAILABLE = False

# --- ARGUMENT PARSING ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="MPC'den QP'ye ve GNN Graf Temsiline veri üretim pipeline'ı.")
    parser.add_argument("--num_train_samples", type=int, default=100,
                        help="Eğitim için üretilecek toplam örnek sayısı.")
    parser.add_argument("--num_test_samples", type=int, default=50,
                        help="Test için üretilecek toplam örnek sayısı.")
    parser.add_argument("--N", type=int, default=10, help="MPC Tahmin Ufku.")
    parser.add_argument("--nx", type=int, default=2, help="Durum değişkenlerinin sayısı.")
    parser.add_argument("--nu", type=int, default=1, help="Kontrol girişlerinin sayısı.")
    parser.add_argument("--umin_val", type=float, default=-1.0, help="Kontrol girişi alt sınırı.")
    parser.add_argument("--umax_val", type=float, default=1.0, help="Kontrol girişi üst sınırı.")
    parser.add_argument("--xmin_val", type=float, default=-5.0, help="Durum değişkenleri alt sınırı.")
    parser.add_argument("--xmax_val", type=float, default=5.0, help="Durum değişkenleri üst sınırı.")
    parser.add_argument("--use_cvxopt", action='store_true',
                        help="QP çözücü olarak CVXOPT kullanılsın mı? (SciPy varsayılandır)")
    parser.add_argument("--force_regen", action='store_true',
                        help="Eğer veri klasörleri varsa, silip yeniden oluşturmaya zorla.")
    return parser.parse_args()

# --- MPC'DEN QP'YE DÖNÜŞÜM ---
def prediction_matrices(A, B, N_horizon):
    nx_dim = A.shape[0]
    nu_dim = B.shape[1]

    F = np.zeros((nx_dim * N_horizon, nx_dim))
    G = np.zeros((nx_dim * N_horizon, nu_dim * N_horizon))

    for i in range(N_horizon):
        F[i * nx_dim:(i + 1) * nx_dim, :] = np.linalg.matrix_power(A, i + 1)
        for j in range(i + 1):
            if i - j >= 0:
                G[i * nx_dim:(i + 1) * nx_dim, j * nu_dim:(j + 1) * nu_dim] = np.linalg.matrix_power(A, i - j) @ B
    return F, G

def form_qp_matrices(F_pred, G_pred, Q_cost_mat, R_cost_mat, P_cost_mat,
                     N_horizon, nx_dim, nu_dim, xmax_val, xmin_val):
    Q_actual_cost = np.eye(nx_dim) * Q_cost_mat[0,0] if Q_cost_mat.ndim == 2 and Q_cost_mat.shape == (1,1) else Q_cost_mat
    P_actual_cost = np.eye(nx_dim) * P_cost_mat[0,0] if P_cost_mat.ndim == 2 and P_cost_mat.shape == (1,1) else P_cost_mat
    R_actual_cost = np.eye(nu_dim) * R_cost_mat[0,0] if R_cost_mat.ndim == 2 and R_cost_mat.shape == (1,1) else R_cost_mat

    Q_blk_diag_list = [Q_actual_cost] * (N_horizon - 1) + [P_actual_cost]
    # Eğer N_horizon 0 veya 1 ise liste boş olabilir veya sadece P_actual_cost içerebilir.
    if not Q_blk_diag_list: # N_horizon=0 ise (anlamsız ama koruma)
        Q_blk_diag = np.array([]).reshape(0,0)
    elif N_horizon == 1: # Sadece P_cost_mat
        Q_blk_diag = P_actual_cost
    else:
        Q_blk_diag = linalg.block_diag(*Q_blk_diag_list)


    R_blk_diag_list = [R_actual_cost] * N_horizon
    if not R_blk_diag_list : # N_horizon=0 ise
         R_blk_diag = np.array([]).reshape(0,0)
    else:
         R_blk_diag = linalg.block_diag(*R_blk_diag_list)


    H_qp = 2 * (G_pred.T @ Q_blk_diag @ G_pred + R_blk_diag)
    H_qp = (H_qp + H_qp.T) / 2

    A_ineq_states_qp = np.vstack((G_pred, -G_pred))

    def calculate_f_qp(x0_current):
        return (2 * G_pred.T @ Q_blk_diag @ F_pred @ x0_current).reshape(-1, 1)

    def calculate_b_ineq_states_qp(x0_current):
        b_upper = (np.tile(xmax_val, N_horizon * nx_dim) - (F_pred @ x0_current).flatten()).reshape(-1, 1)
        b_lower = (-np.tile(xmin_val, N_horizon * nx_dim) + (F_pred @ x0_current).flatten()).reshape(-1, 1)
        return np.vstack((b_upper, b_lower))

    return H_qp, calculate_f_qp, A_ineq_states_qp, calculate_b_ineq_states_qp

# --- QP'DEN GRAF TEMSİLİNE ---
def qp_to_graph_sparse_format(H_mat, f_vec, A_ineq_mat, b_ineq_vec, lb_vec, ub_vec, output_path_instance):
    num_variables = H_mat.shape[0] if H_mat.ndim == 2 else 0
    num_constraints = A_ineq_mat.shape[0] if A_ineq_mat.ndim == 2 else 0

    if num_variables == 0 : # H matrisi boş veya 1D ise (örn N=0)
        var_features_data = np.empty((0,3), dtype=float)
    else:
        var_features_data = np.hstack((f_vec.reshape(num_variables, 1),
                                    lb_vec.reshape(num_variables, 1),
                                    ub_vec.reshape(num_variables, 1)))

    if num_constraints == 0 or b_ineq_vec.size != num_constraints :
        con_features_data = np.empty((0, 2), dtype=float)
    else:
        con_features_data = np.hstack((b_ineq_vec.reshape(num_constraints, 1),
                                       np.zeros((num_constraints, 1))))


    if num_constraints > 0 and A_ineq_mat.size > 0:
        rows_A, cols_A = np.nonzero(A_ineq_mat)
        if rows_A.size > 0:
            A_edges_indices = np.array(list(zip(rows_A, cols_A)))
            A_edge_features_data = A_ineq_mat[rows_A, cols_A].reshape(-1, 1)
        else:
            A_edges_indices = np.empty((0, 2), dtype=int)
            A_edge_features_data = np.empty((0, 1), dtype=float)
    else:
        A_edges_indices = np.empty((0, 2), dtype=int)
        A_edge_features_data = np.empty((0, 1), dtype=float)


    if num_variables > 0 and H_mat.size > 0:
        rows_Q, cols_Q = np.nonzero(H_mat)
        if rows_Q.size > 0:
            Q_edges_indices = np.array(list(zip(rows_Q, cols_Q)))
            Q_edge_features_data = H_mat[rows_Q, cols_Q].reshape(-1, 1)
        else:
            Q_edges_indices = np.empty((0, 2), dtype=int)
            Q_edge_features_data = np.empty((0, 1), dtype=float)
    else:
        Q_edges_indices = np.empty((0, 2), dtype=int)
        Q_edge_features_data = np.empty((0, 1), dtype=float)

    np.savetxt(os.path.join(output_path_instance, "VarFeatures.csv"), var_features_data, delimiter=",", fmt='%10.5f')
    np.savetxt(os.path.join(output_path_instance, "ConFeatures.csv"), con_features_data, delimiter=",", fmt='%10.5f')
    np.savetxt(os.path.join(output_path_instance, "EdgeFeatures_A.csv"), A_edge_features_data, delimiter=",", fmt='%10.5f')
    np.savetxt(os.path.join(output_path_instance, "EdgeIndices_A.csv"), A_edges_indices, delimiter=",", fmt='%d')
    np.savetxt(os.path.join(output_path_instance, "QEdgeFeatures.csv"), Q_edge_features_data, delimiter=",", fmt='%10.5f')
    np.savetxt(os.path.join(output_path_instance, "QEdgeIndices.csv"), Q_edges_indices, delimiter=",", fmt='%d')

# --- RASTGELE SİSTEM ÜRETİMİ ---
def generate_stable_random_system(nx_dim, nu_dim, max_attempts=20):
    A_sys = np.array([]) # Hata durumunda boş array dönmemesi için ilk değer ataması
    for _ in range(max_attempts):
        A_sys_candidate = np.random.uniform(-1.2, 1.2, (nx_dim, nx_dim))
        try:
            eigvals, eigvecs = np.linalg.eig(A_sys_candidate)
            max_abs_eigval = np.max(np.abs(eigvals))

            if np.allclose(A_sys_candidate, 0):
                A_stable_sys = np.zeros((nx_dim,nx_dim))
            elif max_abs_eigval < 1.0 - 1e-9:
                A_stable_sys = A_sys_candidate
            elif max_abs_eigval > 1e-9:
                stabilizing_factor = np.random.uniform(0.6, 0.95) / max_abs_eigval
                scaled_eigvals = eigvals * stabilizing_factor
                A_stable_sys_temp = np.real(eigvecs @ np.diag(scaled_eigvals) @ np.linalg.inv(eigvecs))
                if np.any(np.isnan(A_stable_sys_temp)) or np.any(np.isinf(A_stable_sys_temp)):
                    continue
                A_stable_sys = A_stable_sys_temp
            else:
                A_stable_sys = A_sys_candidate

            final_eigvals_check, _ = np.linalg.eig(A_stable_sys)
            if np.all(np.abs(final_eigvals_check) < 1.0 - 1e-9) or np.allclose(A_stable_sys, 0):
                B_sys = np.random.uniform(-1.0, 1.0, (nx_dim, nu_dim))
                return A_stable_sys, B_sys
            A_sys = A_stable_sys # Son başarılı (veya yarı başarılı) adayı tut
        except np.linalg.LinAlgError:
            A_sys = A_sys_candidate # Hata durumunda son adayı tut
            pass
    print(f"Uyarı: {max_attempts} denemeye rağmen kesin kararlı A üretilemedi. Son aday kullanılıyor.")
    B_sys = np.random.uniform(-1.0, 1.0, (nx_dim, nu_dim))
    return A_sys if A_sys.size > 0 else np.random.rand(nx_dim,nx_dim)*0.1 , B_sys # Eğer A_sys hiç atanmadıysa küçük rastgele ata

# --- QP ÇÖZÜCÜLER VE VERİ ÜRETİM PİPELINE'I ---
def solve_qp_and_get_labels(H_qp, f_qp, A_ineq_qp, b_ineq_qp, lb_qp, ub_qp, solver_type_arg, N_horizon, nu_dim):
    solution_U = None
    is_problem_feasible = False
    optimal_objective_value = None
    num_decision_vars = N_horizon * nu_dim
    
    if num_decision_vars == 0 : # N=0 veya nu=0 ise çözülecek bir şey yok
        is_problem_feasible = True # Boş problem fizibil sayılabilir
        solution_U = np.empty((0,1))
        optimal_objective_value = 0.0
        return is_problem_feasible, solution_U, optimal_objective_value


    if solver_type_arg == 'scipy':
        def cost_fn_scipy(u_flat_vars):
            u_reshaped_vars = u_flat_vars.reshape(-1, 1)
            obj_val = 0.5 * (u_reshaped_vars.T @ H_qp @ u_reshaped_vars) + (f_qp.T @ u_reshaped_vars)
            return obj_val.item()

        def inequality_cons_fn_scipy(u_flat_vars):
            if A_ineq_qp.shape[0] == 0:
                return np.array([])
            return (b_ineq_qp - (A_ineq_qp @ u_flat_vars.reshape(-1, 1))).flatten()

        constraints_for_scipy = []
        if A_ineq_qp.shape[0] > 0:
             constraints_for_scipy.append({'type': 'ineq', 'fun': inequality_cons_fn_scipy})

        bounds_for_scipy = Bounds(lb_qp.flatten(), ub_qp.flatten())
        initial_guess_u = np.clip((lb_qp.flatten() + ub_qp.flatten()) / 2.0,
                                  lb_qp.flatten(), ub_qp.flatten())
        if initial_guess_u.size != num_decision_vars:
             initial_guess_u = np.zeros(num_decision_vars)


        solver_options = {'ftol': 1e-7, 'disp': False, 'maxiter': 3000}
        try:
            result = minimize(cost_fn_scipy, initial_guess_u, method='SLSQP',
                              bounds=bounds_for_scipy, constraints=constraints_for_scipy,
                              options=solver_options)

            if result.success:
                solution_U_candidate = result.x.reshape(-1, 1)
                tol = 1e-5
                cons_ok = True
                if A_ineq_qp.shape[0] > 0:
                    if not np.all((A_ineq_qp @ solution_U_candidate) <= b_ineq_qp + tol):
                        cons_ok = False
                if not (np.all(solution_U_candidate >= lb_qp - tol) and np.all(solution_U_candidate <= ub_qp + tol)):
                    cons_ok = False

                if cons_ok:
                    is_problem_feasible = True
                    optimal_objective_value = result.fun
                    solution_U = solution_U_candidate
            # else: # SciPy bazen success=False verse de yakın bir çözüm bulabilir, kısıt kontrolü önemli.
            #     # print(f"  SciPy: Çözüm başarısız. M: {result.message}, S: {result.status}")
            #     pass
        except Exception:
            pass

    elif solver_type_arg == 'cvxopt' and CVXOPT_AVAILABLE:
        P_cvx = cvxopt.matrix(H_qp)
        q_cvx = cvxopt.matrix(f_qp)

        I_vars = np.eye(num_decision_vars)
        G_control_cvx = np.vstack((I_vars, -I_vars))
        h_control_cvx = np.vstack((ub_qp, -lb_qp))

        if A_ineq_qp.shape[0] > 0:
            G_all_cvx_np = np.vstack((A_ineq_qp, G_control_cvx))
            h_all_cvx_np = np.vstack((b_ineq_qp, h_control_cvx))
        else:
            G_all_cvx_np = G_control_cvx
            h_all_cvx_np = h_control_cvx

        G_cvx_final = cvxopt.matrix(G_all_cvx_np)
        h_cvx_final = cvxopt.matrix(h_all_cvx_np)

        try:
            solution = cvxopt.solvers.qp(P_cvx, q_cvx, G_cvx_final, h_cvx_final)
            if solution['status'] == 'optimal':
                solution_U = np.array(solution['x'])
                is_problem_feasible = True
                optimal_objective_value = solution['primal objective']
            elif solution['status'] == 'unknown':
                u_temp_sol = np.array(solution['x'])
                if u_temp_sol is not None and u_temp_sol.shape[0] == num_decision_vars : # Çözüm var ve boyutu doğru mu?
                    tol = 1e-5
                    if np.all((G_all_cvx_np @ u_temp_sol) <= h_all_cvx_np + tol):
                        solution_U = u_temp_sol
                        is_problem_feasible = True
                        obj_val_recalc = 0.5 * (solution_U.T @ H_qp @ solution_U) + (f_qp.T @ solution_U)
                        optimal_objective_value = obj_val_recalc.item()
        except (ValueError, TypeError, ArithmeticError, Exception): # CVXOPT çeşitli hatalar verebilir
            pass
    else:
        pass

    return is_problem_feasible, solution_U, optimal_objective_value

def generate_dataset(num_total_samples, dataset_basename, config_args):
    base_output_directory = f"./{dataset_basename}_data"
    if config_args.force_regen and os.path.exists(base_output_directory):
        print(f"'{base_output_directory}' siliniyor (force_regen)...")
        shutil.rmtree(base_output_directory)
    os.makedirs(base_output_directory, exist_ok=True)

    print(f"\n--- {dataset_basename.upper()} VERİSİ ÜRETİLİYOR ({num_total_samples} örnek) ---")

    successful_generations = 0
    total_attempts = 0
    max_total_attempts = num_total_samples * 5 # Artırılmış deneme limiti

    start_time_dataset = time.time()

    while successful_generations < num_total_samples and total_attempts < max_total_attempts :
        total_attempts += 1
        instance_id_str = f"Data_{successful_generations}"
        current_instance_path = os.path.join(base_output_directory, instance_id_str)
        # Başarılı olana kadar klasör oluşturma
        # os.makedirs(current_instance_path, exist_ok=True) # Şimdilik kalsın, her deneme için yer açar


        if total_attempts % (max(1, num_total_samples // 10)) == 0 or successful_generations < 3:
             print(f"İlerleme: {successful_generations}/{num_total_samples} (Toplam Deneme: {total_attempts})")

        A_system, B_system = generate_stable_random_system(config_args.nx, config_args.nu)
        if A_system is None : # Kararlı sistem üretilemedi
            # print(f"Deneme {total_attempts}: Kararlı sistem üretilemedi, atlanıyor.")
            continue


        Q_cost_val = np.eye(config_args.nx) * np.random.uniform(0.2, 1.5)
        R_cost_val = np.eye(config_args.nu) * np.random.uniform(0.01, 0.5)
        P_cost_val = Q_cost_val * np.random.uniform(0.8, 1.2)

        F_prediction, G_prediction = prediction_matrices(A_system, B_system, config_args.N)
        # N=0 veya nu=0 gibi durumlar prediction_matrices'te boş matrisler üretebilir,
        # form_qp_matrices bunları ele almalı
        if F_prediction.size == 0 or G_prediction.size == 0 and config_args.N > 0: # N > 0 ise bu olmamalı
            # print(f"Deneme {total_attempts}: F veya G matrisi boş üretildi (N={config_args.N}), atlanıyor.")
            continue


        H_problem, f_problem_func, A_ineq_states_problem, b_ineq_states_problem_func = \
            form_qp_matrices(F_prediction, G_prediction, Q_cost_val, R_cost_val, P_cost_val,
                             config_args.N, config_args.nx, config_args.nu,
                             config_args.xmax_val, config_args.xmin_val)

        lb_controls = np.full((config_args.N * config_args.nu, 1), config_args.umin_val)
        ub_controls = np.full((config_args.N * config_args.nu, 1), config_args.umax_val)

        x0_initial_state = np.random.uniform(config_args.xmin_val * 0.8, config_args.xmax_val * 0.8,
                                             (config_args.nx, 1))
        f_problem_vec = f_problem_func(x0_initial_state)
        b_ineq_states_problem_vec = b_ineq_states_problem_func(x0_initial_state)

        solver_to_use = 'cvxopt' if config_args.use_cvxopt else 'scipy'
        is_feasible_problem, optimal_U_solution, optimal_obj_q_value = \
            solve_qp_and_get_labels(H_problem, f_problem_vec, A_ineq_states_problem,
                                    b_ineq_states_problem_vec, lb_controls, ub_controls,
                                    solver_to_use, config_args.N, config_args.nu)
        
        # Sadece bir şeyler üretildiyse (çözücü hata vermediyse) veya N=0 ise devam et
        if (is_feasible_problem and optimal_U_solution is not None) or not is_feasible_problem or config_args.N == 0:
            # Başarılı fizibil çözüm veya fizibil olmayan durum veya N=0 durumu (boş problem)
            # Bu durumda örneği geçerli sayıp klasörünü oluştur
            os.makedirs(current_instance_path, exist_ok=True)

            pd.DataFrame([[1 if is_feasible_problem else 0]]).to_csv(
                os.path.join(current_instance_path, "Labels_feas.csv"), index=False, header=False)

            if is_feasible_problem and optimal_U_solution is not None and optimal_obj_q_value is not None:
                pd.DataFrame(H_problem).to_csv(os.path.join(current_instance_path, "qp_H.csv"), index=False, header=False)
                pd.DataFrame(f_problem_vec).to_csv(os.path.join(current_instance_path, "qp_f.csv"), index=False, header=False)
                pd.DataFrame(A_ineq_states_problem).to_csv(os.path.join(current_instance_path, "qp_A_ineq.csv"), index=False, header=False)
                pd.DataFrame(b_ineq_states_problem_vec).to_csv(os.path.join(current_instance_path, "qp_b_ineq.csv"), index=False, header=False)
                pd.DataFrame(lb_controls).to_csv(os.path.join(current_instance_path, "qp_lb.csv"), index=False, header=False)
                pd.DataFrame(ub_controls).to_csv(os.path.join(current_instance_path, "qp_ub.csv"), index=False, header=False)
                pd.DataFrame(optimal_U_solution).to_csv(os.path.join(current_instance_path, "Labels_solu.csv"), index=False, header=False)
                pd.DataFrame([[optimal_obj_q_value]]).to_csv(os.path.join(current_instance_path, "Labels_obj.csv"), index=False, header=False)

            # Graf temsili (N=0 ise boş matrisler gidecek, qp_to_graph_sparse_format bunu ele almalı)
            qp_to_graph_sparse_format(H_problem, f_problem_vec if f_problem_vec.size > 0 else np.array([]).reshape(0,1),
                                      A_ineq_states_problem,
                                      b_ineq_states_problem_vec if b_ineq_states_problem_vec.size > 0 else np.array([]).reshape(0,1),
                                      lb_controls, ub_controls,
                                      current_instance_path)
            
            successful_generations += 1 # Sadece başarılı bir şekilde etiketlenen ve grafı oluşturulanları say
        # else:
            # print(f"Deneme {total_attempts}: Çözücü bir sorunla karşılaştı veya fizibil değil, geçerli örnek sayılamadı.")


    end_time_dataset = time.time()
    print(f"--- {dataset_basename.upper()} VERİ ÜRETİMİ TAMAMLANDI ---")
    print(f"İstenen: {num_total_samples}, Üretilen: {successful_generations}, Toplam Deneme: {total_attempts}")
    print(f"Geçen Süre: {end_time_dataset - start_time_dataset:.2f} saniye")


# --- ANA PROGRAM ---
if __name__ == "__main__":
    args = parse_arguments()

    generate_dataset(args.num_train_samples, "train", args)
    generate_dataset(args.num_test_samples, "test", args)

    print("\nTÜM VERİ ÜRETİM PİPELİNE'I TAMAMLANDI.")

    # Colab için indirme komutları (opsiyonel)
    try:
        from IPython import get_ipython
        if 'google.colab' in str(get_ipython()): # type: ignore
            print("\nColab ortamında çalışıyorsunuz. Verileri indirmek için:")
            print("import shutil; shutil.make_archive('train_data_archive', 'zip', 'train_data')")
            print("from google.colab import files; files.download('train_data_archive.zip')")
            print("# Test verisi için benzer şekilde: shutil.make_archive('test_data_archive', 'zip', 'test_data') ve files.download(...)")
    except (NameError, ImportError):
        pass