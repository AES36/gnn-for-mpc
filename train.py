# train_fixed.py (Tamamen temiz versiyon)
import numpy as np
from pandas import read_csv, errors as pd_errors
import pandas as pd
import tensorflow as tf
import argparse
import os
import time
from sklearn.model_selection import train_test_split
from models import QPGNNPolicy

## ARGUMENTS
parser = argparse.ArgumentParser(description="QP GNN Model Training Script - Fixed")
parser.add_argument("--data_folder", help="Training data folder (e.g., ./train_data)", default="./train_data", type=str)
parser.add_argument("--total_samples", help="Total training samples to scan", default=2000, type=int)
parser.add_argument("--val_split", help="Validation split ratio", default=0.2, type=float)
parser.add_argument("--gpu", help="GPU index (-1 for CPU)", default="-1", type=str)
parser.add_argument("--emb_size", help="GNN embedding size", default=32, type=int)
parser.add_argument("--epochs", help="Maximum epochs", default=50, type=int)
parser.add_argument("--type", help="Model type", default="fea", choices=['fea','obj','sol'])
parser.add_argument("--lr", help="Initial learning rate", default=0.001, type=float)
parser.add_argument("--dropout", help="Dropout rate for QPGNNPolicy", default=0.1, type=float)
parser.add_argument("--weight_decay", help="Weight decay for AdamW", default=1e-5, type=float)
parser.add_argument("--batch_size", help="Mini-batch size for training", default=32, type=int)
parser.add_argument("--lr_patience", help="LR reduction patience", default=10, type=int)
parser.add_argument("--lr_factor", help="LR reduction factor", default=0.5, type=float)
parser.add_argument("--min_lr", help="Minimum learning rate for LR scheduler", default=1e-7, type=float)
parser.add_argument("--verbose", help="Verbose output", action="store_true")
parser.add_argument("--model_save_path", help="Model save directory", default="./saved_models", type=str)
parser.add_argument("--N", help="MPC Horizon (for default dim detection)", default=10, type=int)
parser.add_argument("--nx", help="Number of states (for default dim detection)", default=2, type=int)
parser.add_argument("--nu", help="Number of controls (for default dim detection)", default=1, type=int)
args = parser.parse_args()

class AdaptiveLRScheduler:
    def __init__(self, optimizer, patience=10, factor=0.5, min_lr=1e-7, verbose=True):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')
        self.verbose = verbose

    def step(self, val_loss):
        current_lr = self.optimizer.learning_rate.numpy()
        if np.isinf(val_loss) or np.isnan(val_loss):
            if self.verbose: 
                print(f"  ⚠️ LR Scheduler: Geçersiz doğrulama kaybı ({val_loss}), adım atlanıyor.")
            return False
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            old_lr = current_lr
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr > new_lr:
                self.optimizer.learning_rate.assign(new_lr)
                self.wait = 0
                if self.verbose:
                    print(f"  📉 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
                return True
            elif old_lr == self.min_lr:
                self.wait = 0
        return False

class QPModelTrainer:
    def __init__(self, config_args):
        self.args = config_args
        self.train_indices = None
        self.val_indices = None
        self.data_dimensions = {}
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        
    def setup_gpu(self):
        if self.args.gpu == "-1":
            print("🖥️  Running on CPU")
            tf.config.set_visible_devices([], 'GPU')
            return "/CPU:0"
        else:
            gpu_index = int(self.args.gpu)
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 0 and 0 <= gpu_index < len(gpus):
                try:
                    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
                    print(f"🚀 Using GPU {gpu_index}: {gpus[gpu_index].name}")
                    return f"/GPU:{gpu_index}"
                except Exception as e:
                    print(f"⚠️  GPU setup failed: {e}, using CPU")
                    tf.config.set_visible_devices([], 'GPU')
                    return "/CPU:0"
            else:
                print(f"❌ GPU {gpu_index} not found or invalid. Using CPU")
                tf.config.set_visible_devices([], 'GPU')
                return "/CPU:0"

    def detect_data_dimensions(self):
        found_sample = False
        for i in range(self.args.total_samples):
            sample_dir = os.path.join(self.args.data_folder, f"Data_{i}")
            if not os.path.isdir(sample_dir):
                if self.args.verbose and i < 10:
                    print(f"Dim Detect: Skipping Data_{i}, directory not found.")
                continue
            
            try:
                vf_path = os.path.join(sample_dir, "VarFeatures.csv")
                cf_path = os.path.join(sample_dir, "ConFeatures.csv")
                efa_path = os.path.join(sample_dir, "EdgeFeatures_A.csv")
                qefh_path = os.path.join(sample_dir, "QEdgeFeatures.csv")

                if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in [vf_path, cf_path]):
                    if self.args.verbose and i < 10:
                        print(f"Dim Detect: Skipping Data_{i}, Var/ConFeatures missing or empty.")
                    continue

                sample_var = read_csv(vf_path, header=None)
                sample_con = read_csv(cf_path, header=None)
                sample_edge_A = read_csv(efa_path, header=None) if os.path.exists(efa_path) and os.path.getsize(efa_path) > 0 else pd.DataFrame(columns=range(1))
                sample_qedge_H = read_csv(qefh_path, header=None) if os.path.exists(qefh_path) and os.path.getsize(qefh_path) > 0 else pd.DataFrame(columns=range(1))

                self.data_dimensions = {
                    'n_vars': sample_var.shape[0],
                    'n_cons': sample_con.shape[0],
                    'var_features': sample_var.shape[1],
                    'con_features': sample_con.shape[1],
                    'edge_features': sample_edge_A.shape[1] if sample_edge_A.shape[0] > 0 else 0,
                    'qedge_features': sample_qedge_H.shape[1] if sample_qedge_H.shape[0] > 0 else 0,
                    'N_horizon': self.args.N,
                    'nx_dim': self.args.nx,
                    'nu_dim': self.args.nu
                }
                found_sample = True
                print(f"📊 Data dimensions detected from Data_{i}:")
                break 
            except pd_errors.EmptyDataError:
                if self.args.verbose and i < 10:
                    print(f"Dim Detect: Skipping Data_{i}, a CSV file was unexpectedly empty.")
                continue
            except Exception as e:
                if self.args.verbose and i < 10:
                    print(f"❌ Dimension detection for Data_{i} failed: {e}")
                continue

        if not found_sample:
            print(f"❌ Dimension detection failed after {self.args.total_samples} attempts in '{self.args.data_folder}'.")
            print(f"🔧 Using default dimensions based on args: N={self.args.N}, nx={self.args.nx}, nu={self.args.nu}.")
            self.data_dimensions = {
                'n_vars': self.args.N * self.args.nu,
                'n_cons': 2 * self.args.N * self.args.nx,
                'var_features': 3, 'con_features': 2,
                'edge_features': 1, 'qedge_features': 1,
                'N_horizon': self.args.N, 'nx_dim': self.args.nx, 'nu_dim': self.args.nu
            }
        
        print(f"   Variables/graph: {self.data_dimensions['n_vars']}, Constraints/graph: {self.data_dimensions['n_cons']}")

    def create_train_val_split(self):
        available_indices = []
        for i in range(self.args.total_samples):
            instance_dir = os.path.join(self.args.data_folder, f"Data_{i}")
            feas_path = os.path.join(instance_dir, "Labels_feas.csv")
            if not os.path.exists(feas_path):
                if self.args.verbose and i < 10:
                    print(f"Split: Skipping Data_{i}, Labels_feas.csv not found.")
                continue
            try:
                is_feasible = read_csv(feas_path, header=None).values[0,0]
                if self.args.type in ["obj", "sol"] and is_feasible == 0:
                    if self.args.verbose and i < 10:
                        print(f"Split: Skipping Data_{i}, infeasible for task {self.args.type}.")
                    continue
                
                labels_ok = True
                if self.args.type == "obj" and not os.path.exists(os.path.join(instance_dir, "Labels_obj.csv")):
                    labels_ok = False
                elif self.args.type == "sol" and not os.path.exists(os.path.join(instance_dir, "Labels_solu.csv")):
                    labels_ok = False
                if not labels_ok:
                    if self.args.verbose and i < 10:
                        print(f"Split: Skipping Data_{i}, required label for {self.args.type} not found.")
                    continue
                
                required_graph_files = [
                    "VarFeatures.csv", "ConFeatures.csv", 
                    "EdgeIndices_A.csv", "EdgeFeatures_A.csv",
                    "QEdgeIndices.csv", "QEdgeFeatures.csv"
                ]
                all_files_valid = True
                for f_name in required_graph_files:
                    f_path = os.path.join(instance_dir, f_name)
                    if not os.path.exists(f_path):
                        all_files_valid = False
                        break
                    if "Features" in f_name and os.path.getsize(f_path) == 0:
                        is_essential_feature = (f_name == "VarFeatures.csv" and self.data_dimensions.get('var_features',0) > 0) or \
                                             (f_name == "ConFeatures.csv" and self.data_dimensions.get('con_features',0) > 0) or \
                                             (f_name == "EdgeFeatures_A.csv" and self.data_dimensions.get('edge_features',0) > 0) or \
                                             (f_name == "QEdgeFeatures.csv" and self.data_dimensions.get('qedge_features',0) > 0)
                        if is_essential_feature:
                            all_files_valid = False
                            break
                if not all_files_valid:
                    if self.args.verbose and i < 10:
                        print(f"Split: Skipping Data_{i}, one or more essential graph files missing or empty.")
                    continue
                available_indices.append(i)
            except Exception as e:
                if self.args.verbose and i < 10:
                    print(f"Split: Error processing Data_{i}: {e}")
                continue
        
        if len(available_indices) == 0:
            raise ValueError(f"No valid data samples found in '{self.args.data_folder}' for task '{self.args.type}'.")

        if len(available_indices) < 2 and self.args.val_split > 0:
             print(f"Uyarı: Çok az geçerli örnek ({len(available_indices)}). Doğrulama seti oluşturulamadı.")
             self.train_indices = available_indices
             self.val_indices = []
        elif self.args.val_split == 0:
            self.train_indices = available_indices
            self.val_indices = []
        else:
            self.train_indices, self.val_indices = train_test_split(
                available_indices, test_size=self.args.val_split, random_state=42, shuffle=True)

        print(f"📊 Data split: Total {len(available_indices)}, Train {len(self.train_indices)}, Val {len(self.val_indices)}")

    def load_batch_data(self, indices):
        if not indices:
            return None
        varFeatures_list, conFeatures_list, edgFeatures_A_list, edgIndices_A_list, \
        q_edgFeatures_H_list, q_edgIndices_H_list, labels_list = [[] for _ in range(7)]
        var_node_offset, con_node_offset = 0, 0

        for i in indices:
            instance_dir = os.path.join(self.args.data_folder, f"Data_{i}")
            try:
                if self.args.type == "fea":
                    labels_data = np.array([[read_csv(os.path.join(instance_dir, "Labels_feas.csv"), header=None).values[0,0]]], dtype=np.float32)
                elif self.args.type == "obj":
                    labels_data = read_csv(os.path.join(instance_dir, "Labels_obj.csv"), header=None).values.astype(np.float32)
                elif self.args.type == "sol":
                    labels_data = read_csv(os.path.join(instance_dir, "Labels_solu.csv"), header=None).values.astype(np.float32)
                else:
                    continue

                vf = read_csv(os.path.join(instance_dir, "VarFeatures.csv"), header=None).values
                cf = read_csv(os.path.join(instance_dir, "ConFeatures.csv"), header=None).values
                
                efa_path = os.path.join(instance_dir, "EdgeFeatures_A.csv")
                eia_path = os.path.join(instance_dir, "EdgeIndices_A.csv")
                qefh_path = os.path.join(instance_dir, "QEdgeFeatures.csv")
                qeih_path = os.path.join(instance_dir, "QEdgeIndices.csv")

                efa = read_csv(efa_path, header=None).values if os.path.exists(efa_path) and os.path.getsize(efa_path)>0 else np.empty((0,self.data_dimensions['edge_features']))
                eia = read_csv(eia_path, header=None).values if os.path.exists(eia_path) and os.path.getsize(eia_path)>0 else np.empty((0,2))
                qefh = read_csv(qefh_path, header=None).values if os.path.exists(qefh_path) and os.path.getsize(qefh_path)>0 else np.empty((0,self.data_dimensions['qedge_features']))
                qeih = read_csv(qeih_path, header=None).values if os.path.exists(qeih_path) and os.path.getsize(qeih_path)>0 else np.empty((0,2))

                edg_indices_A_offset = eia + [con_node_offset, var_node_offset] if eia.shape[0]>0 else eia
                q_edg_indices_H_offset = qeih + [var_node_offset, var_node_offset] if qeih.shape[0]>0 else qeih

                varFeatures_list.append(vf)
                conFeatures_list.append(cf)
                edgFeatures_A_list.append(efa)
                edgIndices_A_list.append(edg_indices_A_offset)
                q_edgFeatures_H_list.append(qefh)
                q_edgIndices_H_list.append(q_edg_indices_H_offset)
                labels_list.append(labels_data)

                var_node_offset += vf.shape[0]
                con_node_offset += cf.shape[0]
            except Exception as e:
                if self.args.verbose:
                    print(f"⚠️ Failed to load/process Data_{i} in batch: {e}")
                continue
        
        if not varFeatures_list:
            return None

        varFeatures_all = np.vstack(varFeatures_list)
        conFeatures_all = np.vstack(conFeatures_list) if conFeatures_list else np.empty((0, self.data_dimensions['con_features']))
        edgFeatures_A_all = np.vstack(edgFeatures_A_list) if edgFeatures_A_list and any(item.shape[0]>0 for item in edgFeatures_A_list) else np.empty((0,self.data_dimensions['edge_features']))
        edgIndices_A_all = np.vstack(edgIndices_A_list) if edgIndices_A_list and any(item.shape[0]>0 for item in edgIndices_A_list) else np.empty((0,2))
        q_edgFeatures_H_all = np.vstack(q_edgFeatures_H_list) if q_edgFeatures_H_list and any(item.shape[0]>0 for item in q_edgFeatures_H_list) else np.empty((0,self.data_dimensions['qedge_features']))
        q_edgIndices_H_all = np.vstack(q_edgIndices_H_list) if q_edgIndices_H_list and any(item.shape[0]>0 for item in q_edgIndices_H_list) else np.empty((0,2))
        labels_all = np.vstack(labels_list)

        batch_data_tuple = (
            tf.constant(conFeatures_all, dtype=tf.float32),
            tf.transpose(tf.constant(edgIndices_A_all, dtype=tf.int32)),
            tf.constant(edgFeatures_A_all, dtype=tf.float32),
            tf.constant(varFeatures_all, dtype=tf.float32),
            tf.transpose(tf.constant(q_edgIndices_H_all, dtype=tf.int32)),
            tf.constant(q_edgFeatures_H_all, dtype=tf.float32),
            tf.constant(con_node_offset, dtype=tf.int32),
            tf.constant(var_node_offset, dtype=tf.int32),
            tf.constant(self.data_dimensions['n_cons'], dtype=tf.int32),
            tf.constant(self.data_dimensions['n_vars'], dtype=tf.int32),
            tf.constant(labels_all, dtype=tf.float32)
        )
        return batch_data_tuple
    
    def create_model(self):
        output_units = 1
        output_activation_str = None
        if self.args.type == "fea":
            output_activation_str = 'sigmoid'
        elif self.args.type == "sol":
            output_units = self.data_dimensions['n_vars']
            output_activation_str = None
        elif self.args.type == "obj":
            output_units = 1
            output_activation_str = None
        
        self.model = QPGNNPolicy(
            emb_size=self.args.emb_size,
            cons_nfeats=self.data_dimensions['con_features'],
            edge_nfeats=self.data_dimensions['edge_features'],
            var_nfeats=self.data_dimensions['var_features'],
            qedge_nfeats=self.data_dimensions['qedge_features'],
            is_graph_level=(self.args.type != "sol"),
            output_units=output_units,
            output_activation=output_activation_str,
            dropout_rate=self.args.dropout
        )
        
        try:
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.args.lr, 
                weight_decay=self.args.weight_decay
            )
        except AttributeError:
            print("Uyarı: tf.keras.optimizers.AdamW bulunamadı. Adam kullanılıyor.")
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)

        self.lr_scheduler = AdaptiveLRScheduler(
            self.optimizer, 
            patience=self.args.lr_patience, 
            factor=self.args.lr_factor, 
            min_lr=self.args.min_lr, 
            verbose=self.args.verbose
        )
        
        print(f"🔧 Model '{self.args.type}' oluşturuldu. Emb: {self.args.emb_size}, Out: {output_units}")

    def compute_loss(self, y_true, y_pred):
        if self.args.type == "fea":
            # Binary crossentropy - basit versiyon
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        elif self.args.type == "obj":
            # Relative absolute error
            epsilon = 1e-7
            loss = tf.abs(y_true - y_pred) / (tf.abs(y_true) + epsilon)
        elif self.args.type == "sol":
            # Mean squared error
            loss = tf.square(y_true - y_pred)
        else:
            raise ValueError(f"Bilinmeyen tip: {self.args.type}")
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, model_inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(model_inputs, training=True)
            loss = self.compute_loss(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def val_step(self, model_inputs, labels):
        predictions = self.model(model_inputs, training=False)
        loss = self.compute_loss(labels, predictions)
        return loss

    def train(self):
        print("📂 Mini-batch eğitim sistemi başlatılıyor...")
        
        # Model kayıt yolu
        model_base_name = f"qp_{self.args.type}_emb{self.args.emb_size}_N{self.data_dimensions['N_horizon']}nx{self.data_dimensions['nx_dim']}nu{self.data_dimensions['nu_dim']}"
        model_checkpoint_prefix = os.path.join(self.args.model_save_path, model_base_name, "ckpt")
        os.makedirs(os.path.dirname(model_checkpoint_prefix), exist_ok=True)
        
        # Checkpoint oluştur
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        print(f"\n🚀 Eğitim Başlatılıyor: Tip={self.args.type}, Epoch={self.args.epochs}")
        print(f"   Train samples: {len(self.train_indices)}, Batch size: {self.args.batch_size}")
        print(f"   Batches per epoch: {len(self.train_indices) // self.args.batch_size + (1 if len(self.train_indices) % self.args.batch_size != 0 else 0)}")
        print(f"   Model kaydedilecek prefix: {model_checkpoint_prefix}")
        
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            # Training indices'leri karıştır
            np.random.shuffle(self.train_indices)
            
            epoch_train_losses = []
            
            # Mini-batch training
            for batch_start in range(0, len(self.train_indices), self.args.batch_size):
                batch_end = min(batch_start + self.args.batch_size, len(self.train_indices))
                batch_indices = self.train_indices[batch_start:batch_end]
                
                # Batch verilerini yükle
                batch_data = self.load_batch_data(batch_indices)
                if batch_data is None:
                    continue
                
                # Training step
                batch_loss = self.train_step(batch_data[:-1], batch_data[-1])
                epoch_train_losses.append(batch_loss.numpy())
            
            # Epoch train loss ortalaması
            avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')
            
            # Validation loss hesapla
            if self.val_indices:
                val_data = self.load_batch_data(self.val_indices)
                if val_data is not None:
                    val_loss = self.val_step(val_data[:-1], val_data[-1])
                    val_loss_numpy = val_loss.numpy()
                else:
                    val_loss_numpy = float('inf')
            else:
                if epoch == 0:
                    print("Uyarı: Doğrulama seti yok. LR zamanlayıcı eğitim kaybını kullanacak.")
                val_loss_numpy = avg_train_loss

            # Learning rate scheduler
            self.lr_scheduler.step(val_loss_numpy)
            current_lr_val = self.optimizer.learning_rate.numpy()
            epoch_duration = time.time() - epoch_start_time

            print(f"Epoch {epoch+1:4d}/{self.args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss_numpy:.6f} | LR: {current_lr_val:.2e} | Süre: {epoch_duration:.1f}s | Batches: {len(epoch_train_losses)}")
            
            # Her 10 epoch'ta bir veya son epoch'ta modeli kaydet
            if (epoch + 1) % 10 == 0 or epoch + 1 == self.args.epochs:
                try:
                    checkpoint.save(model_checkpoint_prefix)
                    print(f"  ✓ Model ağırlıkları kaydedildi (Epoch {epoch+1})")
                except Exception as e_save:
                    print(f"  ⚠️ Model ağırlıklarını kaydederken hata: {e_save}")

        print("-" * 70)
        print("🎉 Eğitim tamamlandı!")
        print(f"Son model ağırlıkları {model_checkpoint_prefix} adresine kaydedildi.")

    def run(self):
        print("🚀 QP GNN Model Training - Mini-Batch Versiyon")
        print("=" * 60)
        print("Argümanlar:", vars(self.args))

        target_device = self.setup_gpu()
        self.detect_data_dimensions()
        self.create_train_val_split()

        if not self.train_indices:
            print("❌ Eğitim için geçerli örnek bulunamadı. Program sonlandırılıyor.")
            return

        with tf.device(target_device):
            self.create_model()
            self.train()

if __name__ == "__main__":
    trainer = QPModelTrainer(args)
    trainer.run()