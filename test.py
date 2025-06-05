# test_fixed.py - Yeni veri formatına uyarlanmış
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from pandas import read_csv

# Models import
from models import QPGNNPolicy

## ARGUMENTS
parser = argparse.ArgumentParser(description="QP GNN Model Test Script - Fixed")
parser.add_argument("--test_data_folder", help="Test veri klasörü", default="./test_data", type=str)
parser.add_argument("--model_folder", help="Kaydedilmiş modeller klasörü", default="./saved_models", type=str)
parser.add_argument("--num_test_samples", help="Test edilecek örnek sayısı", default=100, type=int)
parser.add_argument("--gpu", help="GPU indeksi (-1 for CPU)", default="0", type=str)
parser.add_argument("--model_type", help="Test edilecek model tipi", default="sol", choices=['fea','obj','sol'])
parser.add_argument("--save_results", help="Sonuçları kaydet", action="store_true")
parser.add_argument("--verbose", help="Detaylı çıktı", action="store_true")
# Model parametreleri
parser.add_argument("--N", help="MPC Horizon", default=10, type=int)
parser.add_argument("--nx", help="Number of states", default=2, type=int)
parser.add_argument("--nu", help="Number of controls", default=1, type=int)
parser.add_argument("--emb_size", help="Model embedding size", default=32, type=int)
args = parser.parse_args()

class QPModelTester:
    def __init__(self):
        self.model = None
        self.test_data = None
        self.results = {}
        
        # Feature dimensions (will be set automatically)
        self.nVarF = None
        self.nConsF = None 
        self.nEdgeF = None
        self.nQEdgeF = None
        self.n_Vars_small = None
        self.n_Cons_small = None
        
    def setup_gpu(self):
        """GPU ayarlarını yap"""
        if args.gpu == "-1":
            print("🖥️  CPU modunda çalışıyor")
            tf.config.set_visible_devices([], 'GPU')
            return "/CPU:0"
        else:
            gpu_index = int(args.gpu)
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 0:
                try:
                    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
                    print(f"🚀 GPU {gpu_index} kullanılıyor: {gpus[gpu_index].name}")
                    return f"/GPU:{gpu_index}"
                except:
                    print("⚠️  GPU ayarı başarısız, CPU kullanılıyor")
                    return "/CPU:0"
            else:
                print("❌ GPU bulunamadı, CPU kullanılıyor")
                return "/CPU:0"
    
    def detect_data_dimensions(self):
        """Veri boyutlarını otomatik algıla"""
        # YENİ FORMAT: Data_0/VarFeatures.csv (Data_0/Data_0/VarFeatures.csv DEĞİL)
        sample_dir = os.path.join(args.test_data_folder, "Data_0")
        try:
            sample_var = read_csv(os.path.join(sample_dir, "VarFeatures.csv"), header=None)
            sample_con = read_csv(os.path.join(sample_dir, "ConFeatures.csv"), header=None)
            sample_edge_A = read_csv(os.path.join(sample_dir, "EdgeFeatures_A.csv"), header=None)
            sample_qedge_H = read_csv(os.path.join(sample_dir, "QEdgeFeatures.csv"), header=None)
            
            self.n_Vars_small = sample_var.shape[0]
            self.n_Cons_small = sample_con.shape[0]
            self.nVarF = sample_var.shape[1]
            self.nConsF = sample_con.shape[1]
            self.nEdgeF = sample_edge_A.shape[1] 
            self.nQEdgeF = sample_qedge_H.shape[1]
            
            print(f"📊 Veri boyutları algılandı:")
            print(f"   Variables per graph: {self.n_Vars_small}")
            print(f"   Constraints per graph: {self.n_Cons_small}")
            print(f"   Variable features: {self.nVarF}")
            print(f"   Constraint features: {self.nConsF}")
            print(f"   Edge features: {self.nEdgeF}")
            print(f"   QEdge features: {self.nQEdgeF}")
            
        except FileNotFoundError as e:
            print(f"❌ Veri boyutu algılama hatası: {e}")
            # Varsayılan değerler - args'dan al
            self.n_Vars_small = args.N * args.nu
            self.n_Cons_small = 2 * args.N * args.nx
            self.nVarF = 3
            self.nConsF = 2
            self.nEdgeF = 1
            self.nQEdgeF = 1
            print("🔧 Varsayılan boyutlar kullanılıyor")
    
    def load_model(self):
        """Kaydedilmiş modeli yükle"""
        device = self.setup_gpu()
        
        with tf.device(device):
            # Model klasörü
            model_base_name = f"qp_{args.model_type}_emb{args.emb_size}_N{args.N}nx{args.nx}nu{args.nu}"
            model_dir = os.path.join(args.model_folder, model_base_name)
            model_path = tf.train.latest_checkpoint(model_dir)
            
            print(f"🔍 Model aranıyor: {model_dir}")
            
            if model_path is None:
                print(f"❌ Checkpoint bulunamadı: {model_dir}")
                # Alternatif path'leri dene
                alt_paths = [
                    os.path.join(args.model_folder, f"qp_{args.model_type}_s{args.emb_size}.pkl"),
                    os.path.join(args.model_folder, f"qp_{args.model_type}_emb{args.emb_size}.ckpt"),
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        print(f"✅ Alternatif model bulundu: {model_path}")
                        break
                else:
                    print("❌ Hiç model bulunamadı!")
                    return False
            else:
                print(f"✅ Checkpoint bulundu: {model_path}")
            
            try:
                # Model parametrelerini ayarla
                output_units = 1
                output_activation = None
                
                if args.model_type == "fea":
                    output_activation = 'sigmoid'
                elif args.model_type == "sol":
                    output_units = self.n_Vars_small
                    output_activation = None
                elif args.model_type == "obj":
                    output_units = 1
                    output_activation = None
                
                # Model oluştur
                self.model = QPGNNPolicy(
                    emb_size=args.emb_size,
                    cons_nfeats=self.nConsF,
                    edge_nfeats=self.nEdgeF,
                    var_nfeats=self.nVarF,
                    qedge_nfeats=self.nQEdgeF,
                    is_graph_level=(args.model_type != "sol"),
                    output_units=output_units,
                    output_activation=output_activation,
                    dropout_rate=0.0
                )
                
                # Dummy forward pass to build model
                dummy_input = (
                    tf.zeros((self.n_Cons_small, self.nConsF)),
                    tf.zeros((2, self.n_Cons_small * self.n_Vars_small), dtype=tf.int32),
                    tf.zeros((self.n_Cons_small * self.n_Vars_small, self.nEdgeF)),
                    tf.zeros((self.n_Vars_small, self.nVarF)),
                    tf.zeros((2, self.n_Vars_small * self.n_Vars_small), dtype=tf.int32),
                    tf.zeros((self.n_Vars_small * self.n_Vars_small, self.nQEdgeF)),
                    tf.constant(self.n_Cons_small, dtype=tf.int32),
                    tf.constant(self.n_Vars_small, dtype=tf.int32),
                    tf.constant(self.n_Cons_small, dtype=tf.int32),
                    tf.constant(self.n_Vars_small, dtype=tf.int32)
                )
                _ = self.model(dummy_input, training=False)
                
                # Model ağırlıklarını yükle
                if model_path.endswith('.pkl'):
                    # Eski format
                    if hasattr(self.model, 'restore_state'):
                        self.model.restore_state(model_path)
                    else:
                        print("⚠️  restore_state metodu bulunamadı")
                        return False
                else:
                    # TensorFlow checkpoint format
                    checkpoint = tf.train.Checkpoint(model=self.model)
                    status = checkpoint.restore(model_path)
                    if hasattr(status, 'expect_partial'):
                        status.expect_partial()
                
                print(f"✅ {args.model_type.upper()} model başarıyla yüklendi!")
                print(f"   Embedding size: {args.emb_size}")
                print(f"   Output units: {output_units}")
                return True
                
            except Exception as e:
                print(f"❌ Model yükleme hatası: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return False
    
    def load_test_data(self):
        """Test verilerini yükle"""
        print(f"📂 Test verileri yükleniyor: {args.num_test_samples} örnek")
        
        varFeatures_list = []
        conFeatures_list = []
        edgFeatures_A_list = []
        edgIndices_A_list = []
        q_edgFeatures_H_list = []
        q_edgIndices_H_list = []
        
        labels_fea = []
        labels_obj = []
        labels_sol = []
        
        var_node_offset = 0
        con_node_offset = 0
        
        loaded_count = 0
        
        for i in range(args.num_test_samples):
            # YENİ FORMAT: Data_i/VarFeatures.csv (Data_i/Data_i/VarFeatures.csv DEĞİL)
            instance_dir = os.path.join(args.test_data_folder, f"Data_{i}")
            
            try:
                # Labels yükle
                feas_label = read_csv(os.path.join(instance_dir, "Labels_feas.csv"), header=None).values[0,0]
                labels_fea.append(feas_label)
                
                if os.path.exists(os.path.join(instance_dir, "Labels_obj.csv")):
                    obj_label = read_csv(os.path.join(instance_dir, "Labels_obj.csv"), header=None).values
                    labels_obj.append(obj_label[0,0])
                else:
                    labels_obj.append(0.0)
                
                if os.path.exists(os.path.join(instance_dir, "Labels_solu.csv")):
                    sol_label = read_csv(os.path.join(instance_dir, "Labels_solu.csv"), header=None).values
                    labels_sol.append(sol_label.flatten())
                else:
                    labels_sol.append(np.zeros(self.n_Vars_small))
                
                # GNN features yükle - YENİ FORMAT
                var_features = read_csv(os.path.join(instance_dir, "VarFeatures.csv"), header=None).values
                con_features = read_csv(os.path.join(instance_dir, "ConFeatures.csv"), header=None).values
                edg_features_A = read_csv(os.path.join(instance_dir, "EdgeFeatures_A.csv"), header=None).values
                edg_indices_A = read_csv(os.path.join(instance_dir, "EdgeIndices_A.csv"), header=None).values
                q_edg_features_H = read_csv(os.path.join(instance_dir, "QEdgeFeatures.csv"), header=None).values
                q_edg_indices_H = read_csv(os.path.join(instance_dir, "QEdgeIndices.csv"), header=None).values
                
                # Offset kenarları
                edg_indices_A_offset = edg_indices_A + [con_node_offset, var_node_offset]
                q_edg_indices_H_offset = q_edg_indices_H + [var_node_offset, var_node_offset]
                
                varFeatures_list.append(var_features)
                conFeatures_list.append(con_features)
                edgFeatures_A_list.append(edg_features_A)
                edgIndices_A_list.append(edg_indices_A_offset)
                q_edgFeatures_H_list.append(q_edg_features_H)
                q_edgIndices_H_list.append(q_edg_indices_H_offset)
                
                var_node_offset += var_features.shape[0]
                con_node_offset += con_features.shape[0]
                loaded_count += 1
                
            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Data_{i} yüklenemedi: {e}")
                continue
        
        if loaded_count == 0:
            print("❌ Hiç test verisi yüklenemedi!")
            return False
        
        # Super-graph oluştur
        varFeatures_all = np.vstack(varFeatures_list)
        conFeatures_all = np.vstack(conFeatures_list)
        edgFeatures_A_all = np.vstack(edgFeatures_A_list)
        edgIndices_A_all = np.vstack(edgIndices_A_list)
        q_edgFeatures_H_all = np.vstack(q_edgFeatures_H_list)
        q_edgIndices_H_all = np.vstack(q_edgIndices_H_list)
        
        # TensorFlow tensörlerine dönüştür
        self.test_data = {
            'batched_states': (
                tf.constant(conFeatures_all, dtype=tf.float32),
                tf.transpose(tf.constant(edgIndices_A_all, dtype=tf.int32)),
                tf.constant(edgFeatures_A_all, dtype=tf.float32),
                tf.constant(varFeatures_all, dtype=tf.float32),
                tf.transpose(tf.constant(q_edgIndices_H_all, dtype=tf.int32)),
                tf.constant(q_edgFeatures_H_all, dtype=tf.float32),
                tf.constant(conFeatures_all.shape[0], dtype=tf.int32),
                tf.constant(varFeatures_all.shape[0], dtype=tf.int32),
                tf.constant(self.n_Cons_small, dtype=tf.int32),
                tf.constant(self.n_Vars_small, dtype=tf.int32)
            ),
            'labels_fea': np.array(labels_fea),
            'labels_obj': np.array(labels_obj),
            'labels_sol': np.array(labels_sol),
            'num_samples': loaded_count
        }
        
        print(f"✅ {loaded_count} test örneği yüklendi")
        return True
    
    def test_model(self):
        """Modeli test et"""
        if not self.test_data or not self.model:
            print("❌ Test verisi veya model yok!")
            return
        
        print(f"\n🧪 {args.model_type.upper()} Model Testi Başlıyor...")
        print(f"{'='*50}")
        
        # Tahmin yap
        start_time = time.time()
        predictions = self.model(self.test_data['batched_states'], training=False)
        inference_time = time.time() - start_time
        
        # Sonuçları değerlendir
        if args.model_type == 'fea':
            self.evaluate_feasibility(predictions, inference_time)
        elif args.model_type == 'obj':
            self.evaluate_objective(predictions, inference_time)
        elif args.model_type == 'sol':
            self.evaluate_solution(predictions, inference_time)
    
    def evaluate_feasibility(self, predictions, inference_time):
        """Feasibility model değerlendirmesi"""
        y_true = self.test_data['labels_fea']
        y_pred_sigmoid = tf.sigmoid(predictions).numpy().flatten()
        y_pred_binary = (y_pred_sigmoid > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        self.results['feasibility'] = {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'predictions': y_pred_sigmoid,
            'true_labels': y_true
        }
        
        print(f"📊 Feasibility Sonuçları:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Inference Time: {inference_time:.4f}s")
        print(f"   Samples per second: {len(y_true)/inference_time:.2f}")
        
        # Distribution
        true_pos = np.sum((y_true == 1) & (y_pred_binary == 1))
        true_neg = np.sum((y_true == 0) & (y_pred_binary == 0))
        false_pos = np.sum((y_true == 0) & (y_pred_binary == 1))
        false_neg = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        print(f"   True Positive: {true_pos}")
        print(f"   True Negative: {true_neg}")
        print(f"   False Positive: {false_pos}")
        print(f"   False Negative: {false_neg}")
    
    def evaluate_objective(self, predictions, inference_time):
        """Objective model değerlendirmesi"""
        y_true = self.test_data['labels_obj']
        y_pred = predictions.numpy().flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Relative error
        relative_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-6)
        mean_relative_error = np.mean(relative_errors)
        
        self.results['objective'] = {
            'mse': mse,
            'mae': mae,
            'relative_error': mean_relative_error,
            'inference_time': inference_time,
            'predictions': y_pred,
            'true_labels': y_true
        }
        
        print(f"📊 Objective Sonuçları:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   Mean Relative Error: {mean_relative_error:.4f} ({mean_relative_error*100:.2f}%)")
        print(f"   Inference Time: {inference_time:.4f}s")
        print(f"   Samples per second: {len(y_true)/inference_time:.2f}")
        
        # Objective range
        print(f"   True Objective Range: [{np.min(y_true):.4f}, {np.max(y_true):.4f}]")
        print(f"   Pred Objective Range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    def evaluate_solution(self, predictions, inference_time):
        """Solution model değerlendirmesi"""
        y_true = np.array(self.test_data['labels_sol'])
        y_pred = predictions.numpy()
        
        num_samples = self.test_data['num_samples']
        
        print(f"🔍 Debug Info:")
        print(f"   y_pred shape: {y_pred.shape}")
        print(f"   num_samples: {num_samples}")
        print(f"   n_Vars_small: {self.n_Vars_small}")
        print(f"   Expected shape: ({num_samples}, {self.n_Vars_small})")
        
        # y_pred reshape
        if len(y_pred.shape) == 2 and y_pred.shape[0] == num_samples:
            # Zaten doğru boyutta
            print("✅ Prediction shape zaten doğru")
        elif len(y_pred.shape) == 2:
            # 2D ama farklı boyut - ilk boyutu kontrol et
            if y_pred.shape[0] == num_samples * self.n_Vars_small:
                # Super-graph format: (1000*10, 10) -> (1000, 10)
                # Her sample için ilk sütunu al
                y_pred_reshaped = []
                for i in range(num_samples):
                    start_idx = i * self.n_Vars_small
                    end_idx = (i + 1) * self.n_Vars_small
                    # Her sample için diagonal elementi al (her variable için kendi prediction)
                    sample_pred = []
                    for j in range(self.n_Vars_small):
                        sample_pred.append(y_pred[start_idx + j, j])
                    y_pred_reshaped.append(sample_pred)
                y_pred = np.array(y_pred_reshaped)
                print("✅ Super-graph prediction reshape yapıldı")
            else:
                print(f"❌ Beklenmeyen 2D shape: {y_pred.shape}")
                return
        else:
            # 1D array - eski mantık
            total_vars = y_pred.shape[0]
            expected_vars = num_samples * self.n_Vars_small
            
            if total_vars >= expected_vars:
                y_pred = y_pred[:expected_vars].reshape(num_samples, self.n_Vars_small)
                print("✅ 1D Prediction reshape yapıldı")
            else:
                print(f"❌ Prediction boyutu yetersiz: {total_vars} < {expected_vars}")
                return
        
        # y_true reshape
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(num_samples, -1)
        if y_true.shape[1] != self.n_Vars_small:
            y_true = y_true[:, :self.n_Vars_small]
        
        print(f"✅ Final shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        
        # Metrics hesapla
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        
        # Per-sample MSE
        sample_mses = []
        for i in range(num_samples):
            sample_mse = mean_squared_error(y_true[i], y_pred[i])
            sample_mses.append(sample_mse)
        
        self.results['solution'] = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'sample_mses': sample_mses,
            'inference_time': inference_time,
            'predictions': y_pred,
            'true_labels': y_true
        }
        
        print(f"📊 Solution Sonuçları:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Per-sample MSE mean: {np.mean(sample_mses):.6f}")
        print(f"   Per-sample MSE std: {np.std(sample_mses):.6f}")
        print(f"   Per-sample RMSE mean: {np.sqrt(np.mean(sample_mses)):.6f}")
        print(f"   Inference Time: {inference_time:.4f}s")
        print(f"   Samples per second: {num_samples/inference_time:.2f}")
        
        # Sample statistics
        print(f"   Solution Range (True): [{np.min(y_true):.4f}, {np.max(y_true):.4f}]")
        print(f"   Solution Range (Pred): [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    def save_results(self):
        """Sonuçları kaydet"""
        if not args.save_results:
            return
        
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for model_name, results in self.results.items():
            # Predictions ve true labels'ı kaydet
            np.save(f"{results_dir}/{model_name}_predictions_{timestamp}.npy", results['predictions'])
            np.save(f"{results_dir}/{model_name}_true_labels_{timestamp}.npy", results['true_labels'])
            
            # Metrics'i kaydet
            metrics = {k: v for k, v in results.items() if k not in ['predictions', 'true_labels']}
            pd.DataFrame([metrics]).to_csv(f"{results_dir}/{model_name}_metrics_{timestamp}.csv", index=False)
        
        print(f"💾 Sonuçlar kaydedildi: {results_dir}/")
    
    def run_test(self):
        """Ana test fonksiyonu"""
        print("🚀 QP GNN Model Test Script - Fixed Version")
        print("="*60)
        print(f"Model Type: {args.model_type}")
        print(f"Test Samples: {args.num_test_samples}")
        print(f"Embedding Size: {args.emb_size}")
        
        # Boyutları algıla
        self.detect_data_dimensions()
        
        # Modeli yükle
        if not self.load_model():
            print("❌ Model yüklenemedi!")
            return
        
        # Test verilerini yükle
        if not self.load_test_data():
            print("❌ Test verileri yüklenemedi!")
            return
        
        # Testi çalıştır
        self.test_model()
        
        # Sonuçları kaydet
        self.save_results()
        
        print(f"\n🎉 Test tamamlandı!")
        print(f"📊 {args.model_type.upper()} modeli {self.test_data['num_samples']} örnekle test edildi")

if __name__ == "__main__":
    tester = QPModelTester()
    tester.run_test()