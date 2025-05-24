import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy import linalg
import os
import shutil
# import zipfile # Zip kütüphanesi kaldırıldı
import argparse

# cvxopt kütüphanesinin yüklenip yüklenmediğini kontrol et
try:
    import cvxopt
    import cvxopt.solvers
    cvxopt.solvers.options['show_progress'] = False # Sessiz mod
    CVXOPT_AVAILABLE = True
except ImportError:
    print("Uyarı: cvxopt yüklü değil. Sadece SciPy çözücü kullanılabilir.")
    CVXOPT_AVAILABLE = False


# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="MPC'den QP'ye ve GNN Graf Temsiline veri üretim pipeline'ı.")
parser.add_argument("--num_train_samples", type=int, default=100,
                    help="Eğitim için üretilecek toplam örnek sayısı (feasible/infeasible karışık).")
parser.add_argument("--num_test_samples", type=int, default=20,
                    help="Test için üretilecek toplam örnek sayısı (feasible/infeasible karışık).")
parser.add_argument("--N", type=int, default=10, help="MPC Tahmin Ufku.")
parser.add_argument("--nx", type=int, default=2, help="Durum değişkenlerinin sayısı.")
parser.add_argument("--nu", type=int, default=1, help="Kontrol girişlerinin sayısı.")
parser.add_argument("--ny", type=int, default=1, help="Çıkışların sayısı.")
parser.add_argument("--Tsim", type=int, default=5,
                    help="Her bir MPC simülasyonunun zaman adımı sayısı (QP verisi 0. adımda alınır).")
parser.add_argument("--umin_val", type=float, default=-1.0, help="Kontrol girişi alt sınırı.")
parser.add_argument("--umax_val", type=float, default=1.0, help="Kontrol girişi üst sınırı.")
parser.add_argument("--xmin_val", type=float, default=-5.0, help="Durum değişkenleri alt sınırı.")
parser.add_argument("--xmax_val", type=float, default=5.0, help="Durum değişkenleri üst sınırı.")
parser.add_argument("--use_cvxopt", action='store_true',
                    help="QP çözücü olarak CVXOPT kullanılsın mı? (SciPy varsayılandır)")

args = parser.parse_args()


# --- MPC'den QP'ye Dönüşüm Fonksiyonları ---
def prediction_matrices(A, B, N):
    """Durum tahmin matrislerini oluşturur."""
    nx = A.shape[0]
    nu = B.shape[1]
    F = np.zeros((nx * N, nx))
    for i in range(N): # 0'dan N-1'e gitmeli
        F[i * nx:(i + 1) * nx, :] = np.linalg.matrix_power(A, i + 1)
    G = np.zeros((nx * N, nu * N))
    for i in range(N): # 0'dan N-1'e gitmeli
        for j in range(i + 1):
            G[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = np.linalg.matrix_power(A, i - j) @ B
    return F, G


def form_qp(F, G, Q_cost, R_cost, P_cost, umin_val, umax_val, xmin_val, xmax_val, N, nx, nu, ny):
    """
    MPC problem matrislerinden QP problem matrislerini (H, f_x0, A_ineq, b_ineq) oluşturur.
    Tüm kısıtlar (durum ve kontrol) A_ineq ve b_ineq içinde birleştirilir.
    """
    # Q_cost ve P_cost artık (nx, nx) boyutunda olmalı
    # Eğer Q_cost veya P_cost tek boyutlu (skaler) ise, bunu (nx, nx) boyutunda birim matrise dönüştür.
    Q_cost_nx = np.eye(nx) * Q_cost[0,0] if Q_cost.ndim == 2 and Q_cost.shape == (1,1) else Q_cost
    P_cost_nx = np.eye(nx) * P_cost[0,0] if P_cost.ndim == 2 and P_cost.shape == (1,1) else P_cost
    
    # Q_blk: prediction horizon boyunca her adım için Q_cost, son adım için P_cost
    # H'nin formuna uyacak şekilde (N*nx, N*nx) boyutunda olmalı
    Q_blk = linalg.block_diag(*([Q_cost_nx] * (N - 1)), P_cost_nx) # Boyut: (N*nx, N*nx)

    R_cost_nu = np.eye(nu) * R_cost[0,0] if R_cost.ndim == 2 and R_cost.shape == (1,1) else R_cost
    R_blk = linalg.block_diag(*([R_cost_nu] * N))             # Boyut: (N*nu, N*nu)
    H = 2 * (G.T @ Q_blk @ G + R_blk)             # Boyut: (N*nu, N*nu)
    H = (H + H.T) / 2 # Simetrik hale getir

    def f_x0(x0_current): # x0'ı x0_current olarak adlandırıldı, karışıklığı önlemek için
        return 2 * G.T @ Q_blk @ F @ x0_current      # Boyut: (N*nu, 1)

    # Durum kısıtlamalarının A matrisi kısmı
    A_state_pos = G # Boyut: (N*nx, N*nu)
    A_state_neg = -G # Boyut: (N*nx, N*nu)

    # Kontrol girdisi kısıtlamalarının A matrisi kısmı (I ve -I matrisleri)
    A_control_pos = np.eye(N * nu) # Boyut: (N*nu, N*nu)
    A_control_neg = -np.eye(N * nu) # Boyut: (N*nu, N*nu)

    # Genel A_ineq matrisi: Durum ve kontrol kısıtları birleştirilir
    A_ineq_combined = np.vstack((A_state_pos, A_state_neg, A_control_pos, A_control_neg))
    # Toplam satır: 2*N*nx + 2*N*nu
    # Sütun: N*nu

    def bineq_x0(x0_current): # x0'ı x0_current olarak adlandırıldı
        # Durum kısıtlarının b vektörü kısmı
        # xmin_val ve xmax_val tekil değerler olduğundan, np.tile bunları doğru şekilde yayınlar
        b_max_states = np.tile(xmax_val, (N * nx, 1)) - F @ x0_current # Boyut: (N*nx, 1)
        b_min_states = -np.tile(xmin_val, (N * nx, 1)) + F @ x0_current # Boyut: (N*nx, 1)

        # Kontrol girdisi kısıtlarının b vektörü kısmı
        # umin_val ve umax_val tekil değerler olduğundan
        b_max_controls = np.tile(umax_val, (N * nu, 1)) # Boyut: (N*nu, 1)
        b_min_controls = -np.tile(umin_val, (N * nu, 1)) # Boyut: (N*nu, 1)

        # Genel b_ineq vektörü: Durum ve kontrol kısıtları birleştirilir
        return np.vstack((b_max_states, b_min_states, b_max_controls, b_min_controls))
        # Toplam satır: 2*N*nx + 2*N*nu
        # Sütun: 1

    # GNN'in VarFeatures'ında kullanılacak lb ve ub değerleri için
    lb_vec_u = np.tile(umin_val, (N * nu, 1))
    ub_vec_u = np.tile(umax_val, (N * nu, 1))
    
    return H, f_x0, A_ineq_combined, bineq_x0, lb_vec_u, ub_vec_u


# --- QP'yi GNN Graf Temsiline Çevirme Fonksiyonu (Yoğun Format) ---
def qp_to_graph_dense_format(H, f_vec, A_ineq, b_ineq, lb, ub, output_folder, data_id):
    """
    Belirli bir QP örneğini GNN'e uygun yoğun graf temsil formatına dönüştürür
    ve belirtilen çıktı klasörüne kaydeder.

    Args:
        H (numpy.ndarray): Hessian matrisi (n x n).
        f_vec (numpy.ndarray): Doğrusal terim vektörü (n x 1).
        A_ineq (numpy.ndarray): Eşitsizlik kısıt matrisi (m x n).
        b_ineq (numpy.ndarray): Eşitsizlik kısıt vektörü (m x 1).
        lb (numpy.ndarray): Alt sınırlar (n x 1). (Değişken özelliklerinde kullanılacak)
        ub (numpy.ndarray): Üst sınırlar (n x 1). (Değişken özelliklerinde kullanılacak)
        output_folder (str): Çıktı dosyalarının kaydedileceği klasör yolu.
        data_id (int): Veri örneği kimliği (dosya adları için).
    """
    n = H.shape[0]  # Değişken sayısı (N * nu)
    m = A_ineq.shape[0]  # Eşitsizlik kısıt sayısı (2*N*nx + 2*N*nu)

    # Değişken Düğüm Özellikleri (VarFeatures)
    # Şekil: (n, 3) -> [f_değeri, lb_değeri, ub_değeri]
    var_features = np.hstack((f_vec.reshape(n, 1), lb.reshape(n, 1), ub.reshape(n, 1)))

    # Kısıt Düğüm Özellikleri (ConFeatures)
    # Şekil: (m, 2) -> [b_ineq_değeri, kısıt_tipi (0 for <=, 1 for ==)]
    # MPC'de genellikle sadece <= kısıtları vardır, bu yüzden ikinci sütun 0.
    con_features = np.hstack((b_ineq.reshape(m, 1), np.zeros((m, 1))))

    # A_ineq için Kenar Özellikleri ve İndeksleri (Kısıt-Değişken Bağlantıları) - SIFIRLAR DAHİL
    A_edges = []
    A_edge_features = []
    for i in range(m):
        for j in range(n):
            A_edges.append([i, j])  # Kısıt i'den değişken j'ye
            A_edge_features.append(A_ineq[i, j])
    A_edges = np.array(A_edges)  # Şekil: (m*n, 2)
    A_edge_features = np.array(A_edge_features).reshape(-1, 1)  # Şekil: (m*n, 1)

    # H için Kenar Özellikleri ve İndeksleri (Değişken-Değişken Bağlantıları) - SIFIRLAR DAHİL
    Q_edges = []
    Q_edge_features = []
    for i in range(n):
        for j in range(n):
            Q_edges.append([i, j])  # Değişken i'den değişken j'ye
            Q_edge_features.append(H[i, j])
    Q_edges = np.array(Q_edges)  # Şekil: (n*n, 2)
    Q_edge_features = np.array(Q_edge_features).reshape(-1, 1)  # Şekil: (n*n, 1)

    # Çıktı alt klasörünü oluştur
    instance_dir = os.path.join(output_folder, f"Data_{data_id}")
    os.makedirs(instance_dir, exist_ok=True)

    # CSV dosyalarına kaydet
    np.savetxt(os.path.join(instance_dir, "VarFeatures.csv"), var_features, delimiter=",", fmt='%10.5f')
    np.savetxt(os.path.join(instance_dir, "ConFeatures.csv"), con_features, delimiter=",", fmt='%10.5f')
    np.savetxt(os.path.join(instance_dir, "EdgeFeatures_A.csv"), A_edge_features, fmt='%10.5f')
    np.savetxt(os.path.join(instance_dir, "EdgeIndices_A.csv"), A_edges, delimiter=",", fmt='%d')
    np.savetxt(os.path.join(instance_dir, "QEdgeFeatures.csv"), Q_edge_features, fmt='%10.5f')
    np.savetxt(os.path.join(instance_dir, "QEdgeIndices.csv"), Q_edges, delimiter=",", fmt='%d')

    print(f"  Graf temsili Data_{data_id} klasörüne kaydedildi.")


# --- MPC Veri Üretim Pipeline'ı ---
def generate_mpc_qp_graph_data(num_samples, dataset_type, args):
    """
    Belirtilen sayıda MPC'den türetilmiş QP ve Graf verisi üretir.

    Args:
        num_samples (int): Üretilecek toplam örnek sayısı.
        dataset_type (str): 'train' veya 'test' (klasör yapısı için).
        args (argparse.Namespace): Komut satırı argümanları.
    """
    output_base_dir = f"./{dataset_type}_data"
    # Önceki verileri temizle
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"\n--- {dataset_type.upper()} Verisi Üretiliyor ({num_samples} örnek) ---")

    generated_count = 0
    attempt_count = 0

    while generated_count < num_samples:
        attempt_count += 1
        print(f"\nÖrnek {generated_count + 1}/{num_samples} (Deneme: {attempt_count})")

        # Rastgele kararlı sistem oluştur
        A, B, C = generate_stable_random_system(args.nx, args.nu, args.ny)

        # MPC parametreleri
        N = args.N
        # ÖNEMLİ DÜZELTME: Q ve P matrisleri (nx,nx) boyutunda olmalı,
        # çünkü H ve f fonksiyonlarındaki Q_blk doğrudan durumlar üzerine uygulanıyor.
        # Basitlik için, Q ve P'yi nx boyutunda birim matrislerin ağırlıklı versiyonu yapalım.
        Q_cost = np.eye(args.nx) * np.random.uniform(0.1, 1.0) # Durum ağırlık matrisi
        R_cost = np.eye(args.nu) * np.random.uniform(0.01, 0.1) # Kontrol ağırlık matrisi
        P_cost = np.eye(args.nx) * np.random.uniform(0.1, 1.0) # Terminal ağırlık matrisi

        # Kısıtlamalar (tekil değerler olarak tanımlanır, sonra matris olarak tile edilecek)
        umin_val = args.umin_val
        umax_val = args.umax_val
        xmin_val = args.xmin_val
        xmax_val = args.xmax_val

        try:
            # form_qp'ye Q_cost, R_cost, P_cost ve tekil kısıt değerlerini geçirin
            F_mat, G_mat = prediction_matrices(A, B, N) # F ve G matrislerini burada hesapla
            H, f_x0_func, A_ineq_combined, bineq_x0_func, lb_vec_u, ub_vec_u = \
                form_qp(F_mat, G_mat, Q_cost, R_cost, P_cost, umin_val, umax_val, xmin_val, xmax_val, N, args.nx, args.nu, args.ny)

            # Rastgele bir başlangıç durumu seç
            x0 = np.random.uniform(args.xmin_val * 0.5, args.xmax_val * 0.5, (args.nx, 1))

            # f ve b_ineq vektörlerini mevcut x0 için hesapla
            f_vec = f_x0_func(x0)
            b_ineq_vec = bineq_x0_func(x0) # Bu artık 2*N*nx + 2*N*nu boyutunda olmalı

            # QP çözümü altyapısı (fizibilite kontrolü için)
            solver_type = 'cvxopt' if args.use_cvxopt and CVXOPT_AVAILABLE else 'scipy'
            U_solution = None
            is_feasible = False
            optimal_obj_val = None

            if solver_type == 'scipy':
                def cost_fun_scipy(u_flat):
                    u_reshaped = u_flat.reshape(-1, 1)
                    return (0.5 * (u_reshaped.T @ H @ u_reshaped) + (f_vec.T @ u_reshaped)).item()

                def inequality_constraints_fun(u_flat):
                    return b_ineq_vec.flatten() - (A_ineq_combined @ u_flat.reshape(-1, 1)).flatten()

                initial_guess = np.zeros(N * args.nu) # U'nun boyutu N*nu

                constraints_list = [{'type': 'ineq', 'fun': inequality_constraints_fun}]

                # SciPy.optimize.minimize'da Bounds parametresi ayrı verilir
                # Bu yüzden lb_vec_u ve ub_vec_u direkt burada kullanılır
                bounds_scipy_solver = Bounds(lb_vec_u.flatten(), ub_vec_u.flatten())

                result = minimize(cost_fun_scipy, initial_guess, method='SLSQP',
                                  bounds=bounds_scipy_solver, constraints=constraints_list,
                                  options={'ftol': 1e-6, 'disp': False, 'maxiter': 1000})

                if result.success and result.status == 0: # 0 for success (SLSQP)
                    U_solution = result.x.reshape(-1, 1)
                    is_feasible = True
                    optimal_obj_val = result.fun
                    print(f"  SciPy çözücü: Fizibil ve Optimal bulundu. Durum: {result.message}")
                else:
                    print(f"  SciPy çözücü: Fizibil değil veya hata. Durum: {result.message}")
                    is_feasible = False

            elif solver_type == 'cvxopt':
                if not CVXOPT_AVAILABLE:
                    print("  CVXOPT yüklü değil, SciPy kullanılacak (eğer ayarlanmışsa).")
                    is_feasible = False
                else:
                    P_cvx = cvxopt.matrix(H)
                    q_cvx = cvxopt.matrix(f_vec)
                    
                    G_all_cvx = cvxopt.matrix(A_ineq_combined)
                    h_all_cvx = cvxopt.matrix(b_ineq_vec)

                    sol = cvxopt.solvers.qp(P_cvx, q_cvx, G_all_cvx, h_all_cvx)

                    if sol['status'] == 'optimal':
                        U_solution = np.array(sol['x']).reshape(-1, 1)
                        is_feasible = True
                        optimal_obj_val = sol['primal objective'] # CVXOPT'tan hedef değeri al
                        print(f"  CVXOPT çözücü: Fizibil ve Optimal bulundu. Durum: {sol['status']}")
                    else:
                        print(f"  CVXOPT çözücü: Fizibil değil veya hata. Durum: {sol['status']}")
                        is_feasible = False
            else:
                print("  Geçerli bir QP çözücü bulunamadı veya seçilmedi.")
                is_feasible = False

            # Ortak Kayıt Bloğu
            instance_output_dir = os.path.join(output_base_dir, f"Data_{generated_count}")
            os.makedirs(instance_output_dir, exist_ok=True)

            if is_feasible:
                # QP verilerini (H, f, A_ineq_combined, b_ineq_vec, lb_vec_u, ub_vec_u) kaydet
                # header=False, index=False ile kaydedilir.
                pd.DataFrame(H).to_csv(os.path.join(instance_output_dir, "qp_hessian.csv"), index=False, header=False)
                pd.DataFrame(f_vec).to_csv(os.path.join(instance_output_dir, "qp_f_vector.csv"), index=False, header=False)
                pd.DataFrame(A_ineq_combined).to_csv(os.path.join(instance_output_dir, "qp_Aineq.csv"), index=False, header=False)
                pd.DataFrame(b_ineq_vec).to_csv(os.path.join(instance_output_dir, "qp_bineq.csv"), index=False, header=False)
                pd.DataFrame(lb_vec_u).to_csv(os.path.join(instance_output_dir, "qp_lb.csv"), index=False, header=False)
                pd.DataFrame(ub_vec_u).to_csv(os.path.join(instance_output_dir, "qp_ub.csv"), index=False, header=False)
                pd.DataFrame([[1]]).to_csv(os.path.join(instance_output_dir, "Labels_feas.csv"), index=False, header=False)
                pd.DataFrame(U_solution).to_csv(os.path.join(instance_output_dir, "Labels_solu.csv"), index=False, header=False) # Optimal çözüm
                pd.DataFrame([[optimal_obj_val]]).to_csv(os.path.join(instance_output_dir, "Labels_obj.csv"), index=False, header=False) # Optimal hedef değeri

                # QP verilerini GNN'e uygun graf temsil formatına dönüştür (dense)
                qp_to_graph_dense_format(H, f_vec, A_ineq_combined, b_ineq_vec, lb_vec_u, ub_vec_u,
                                        instance_output_dir, generated_count)
                generated_count += 1
            else:
                # Fizibil olmayan veya çözülemeyen örnekler için flag'i kaydet
                pd.DataFrame([[0]]).to_csv(os.path.join(instance_output_dir, "Labels_feas.csv"), index=False, header=False)
                # Fizibil değilse solu ve obj dosyaları olmaz (GNN eğitiminde bu bilgi kullanılır)

        except Exception as e:
            print(f"  Hata oluştu, örnek atlandı: {e}")
            # Hata durumunda da fizibilite flag'ini 0 olarak kaydet
            instance_output_dir = os.path.join(output_base_dir, f"Data_{generated_count}")
            os.makedirs(instance_output_dir, exist_ok=True)
            pd.DataFrame([[0]]).to_csv(os.path.join(instance_output_dir, "Labels_feas.csv"), index=False, header=False)


# --- Rastgele Kararlı Sistem Oluşturma Fonksiyonu ---
def generate_stable_random_system(nx=2, nu=1, ny=1):
    """Kararlı rastgele sistem oluşturur."""
    A = np.random.uniform(-1, 1, (nx, nx))
    eigvals, eigvecs = np.linalg.eig(A) # Özdeğerleri ve özvektörleri birlikte al
    
    # Kararlılık için öz değerlerin mutlak değeri 1'den küçük olmasını sağlar
    max_abs_eigval = np.max(np.abs(eigvals))
    if max_abs_eigval > 0: # Sıfıra bölme hatasını engelle
        stabilizing_factor = np.random.uniform(0.5, 0.9) / max_abs_eigval
        scaled_eigvals = eigvals * stabilizing_factor
        # A matrisini özvektörleri ve ölçeklenmiş özdeğerleri kullanarak yeniden oluştur
        A = np.real(eigvecs @ np.diag(scaled_eigvals) @ np.linalg.inv(eigvecs))
    else: # Eğer max_abs_eigval 0 ise, A zaten sıfır matrisine yakın demektir
        # Bu durumda stabil bir sistem için A'yı sıfır matrisi veya çok küçük değerler olarak ayarla
        A = np.zeros((nx,nx)) # Veya np.random.uniform(-0.01, 0.01, (nx, nx)) gibi

    B = np.random.uniform(-1, 1, (nx, nu))
    C = np.random.uniform(-1, 1, (ny, nx))
    return A, B, C


# --- Ana Program ---
if __name__ == "__main__":
    # Eğitim verisi üretimi
    generate_mpc_qp_graph_data(args.num_train_samples, "train", args)

    # Test verisi üretimi
    generate_mpc_qp_graph_data(args.num_test_samples, "test", args)

    print("\n--- Veri Üretim Pipeline'ı Tamamlandı ---")

    # Zip sıkıştırma kaldırıldı
    # Dosyalar doğrudan train_data ve test_data klasörlerine kaydedildi.

    # Colab için indirme kodu (eğer Colab'de çalışıyorsanız)
    # Bu kısım sadece Colab ortamında çalışırken kullanın
    try:
        from IPython import get_ipython # get_ipython'ı buradan içe aktarın
        if 'google.colab' in str(get_ipython()):
            # Colab'de üretilen klasörleri zipleyip indirme seçeneği sunmak hala faydalı olabilir
            print("\nColab ortamında olduğunuz için, klasörleri ZIPleyip indirmek için aşağıdaki adımları kullanabilirsiniz:")
            print("import shutil")
            print("shutil.make_archive('train_data_zip', 'zip', 'train_data')")
            print("shutil.make_archive('test_data_zip', 'zip', 'test_data')")
            print("from google.colab import files")
            print("files.download('train_data_zip.zip')")
            print("files.download('test_data_zip.zip')")
    except NameError:
        pass # get_ipython tanımlı değilse (yani Colab'de değilse) hata vermez