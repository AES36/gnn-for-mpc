import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
# QPGNNPolicy modelinizi içeren dosya adını doğru bir şekilde buraya girin.
from models import QPGNNPolicy

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser(description="QP GNN Modelini Eğitme Betiği.")
parser.add_argument("--data_folder", help="Eğitim veri setinin ana klasörü", default="./train_data", type=str)
parser.add_argument("--val_data_folder", help="Doğrulama veri setinin ana klasörü", default="./test_data", type=str)
parser.add_argument("--total_samples", help="Toplam eğitim + doğrulama örnek sayısı", default=100, type=int)
parser.add_argument("--num_train_samples", help="Eğitim için kullanılacak örnek sayısı", default=80, type=int)
parser.add_argument("--num_test_samples", help="Test/doğrulama için kullanılacak örnek sayısı", default=20, type=int)
parser.add_argument("--gpu", help="Kullanılacak GPU indeksi", default="0", type=str)
parser.add_argument("--embSize", help="GNN'nin gömme boyutu", default=64, type=int)
parser.add_argument("--epoch", help="Maksimum epoch sayısı", default=500, type=int)
parser.add_argument("--type", help="Model tipi: 'fea' (fizibilite), 'obj' (amaç değeri), 'sol' (çözüm)", default="fea", choices=['fea','obj','sol'])
parser.add_argument("--valSplit", help="Doğrulama için verinin oranı (0~1)", default=0.2, type=float)
parser.add_argument("--dropout", help="Dropout oranı", default=0.0, type=float)
parser.add_argument("--weightDecay", help="AdamW için ağırlık azaltma oranı", default=0.0, type=float)
parser.add_argument("--patience", help="Erken durdurma sabrı", default=300, type=int)
parser.add_argument("--N", help="Zaman ufku uzunluğu", default=10, type=int)
parser.add_argument("--nx", help="Durum değişkeni sayısı", default=2, type=int)
parser.add_argument("--nu", help="Kontrol değişkeni sayısı", default=1, type=int)
args = parser.parse_args()

## HELPER FUNCTIONS
def relative_loss(y_true, y_pred):
    epsilon = 1e-6
    return tf.reduce_mean(tf.abs(y_true - y_pred) / (tf.abs(y_true) + epsilon))

def solution_mse_loss(y_true, y_pred):
    """
    Solution modeli için basit MSE loss.
    Super-graph sorununu MSE ile çözüyoruz.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

def load_and_batch_data(folder_path, n_samples, n_Cons_small, n_Vars_small, load_solution_labels=True, model_type='fea'):
    varFeatures_list = []
    conFeatures_list = []
    edgFeatures_A_list = []
    edgIndices_A_list = []
    q_edgFeatures_H_list = []
    q_edgIndices_H_list = []
    labels_list = []

    var_node_offset = 0
    con_node_offset = 0

    for i in range(n_samples):
        instance_dir = os.path.join(folder_path, f"Data_{i}")
        
        # Fizibilite etiketini kontrol et
        feas_path = os.path.join(instance_dir, "Labels_feas.csv")
        if not os.path.exists(feas_path):
            print(f"Uyarı: {feas_path} bulunamadı, örnek {i} atlanıyor.")
            continue
        
        is_feasible = read_csv(feas_path, header=None).values[0,0]

        # Eğer model 'obj' veya 'sol' ise ve problem fizibil değilse atla
        if model_type in ["obj", "sol"] and is_feasible == 0:
            continue
        
        # Label'ları oku
        if model_type == "fea":
            labels_data = np.array([[is_feasible]])
        elif model_type == "obj":
            obj_path = os.path.join(instance_dir, "Labels_obj.csv")
            if not os.path.exists(obj_path):
                continue 
            labels_data = read_csv(obj_path, header=None).values
        elif model_type == "sol":
            solu_path = os.path.join(instance_dir, "Labels_solu.csv")
            if not os.path.exists(solu_path):
                continue
            labels_data = read_csv(solu_path, header=None).values

        # GNN dosyaları iç Data_X klasöründe
        gnn_data_dir = os.path.join(instance_dir, f"Data_{i}")
        
        # Verileri oku
        try:
            var_features = read_csv(os.path.join(gnn_data_dir, "VarFeatures.csv"), header=None).values
            con_features = read_csv(os.path.join(gnn_data_dir, "ConFeatures.csv"), header=None).values
            edg_features_A = read_csv(os.path.join(gnn_data_dir, "EdgeFeatures_A.csv"), header=None).values
            edg_indices_A = read_csv(os.path.join(gnn_data_dir, "EdgeIndices_A.csv"), header=None).values
            q_edg_features_H = read_csv(os.path.join(gnn_data_dir, "QEdgeFeatures.csv"), header=None).values
            q_edg_indices_H = read_csv(os.path.join(gnn_data_dir, "QEdgeIndices.csv"), header=None).values
        except FileNotFoundError as e:
            print(f"Hata: {e}. Örnek {i} için dosya bulunamadı, atlanıyor.")
            continue
        
        # Boyut kontrolleri
        if (var_features.shape[0] != n_Vars_small or 
            con_features.shape[0] != n_Cons_small or 
            edg_indices_A.shape[0] != n_Cons_small * n_Vars_small or
            q_edg_indices_H.shape[0] != n_Vars_small * n_Vars_small):
            print(f"Uyarı: Örnek {i} için beklenen boyutlar eşleşmiyor, atlanıyor.")
            print(f"Beklenen: var_feats={n_Vars_small}, con_feats={n_Cons_small}, A_edges={n_Cons_small*n_Vars_small}, H_edges={n_Vars_small*n_Vars_small}")
            print(f"Bulunan: var_feats={var_features.shape[0]}, con_feats={con_features.shape[0]}, A_edges={edg_indices_A.shape[0]}, H_edges={q_edg_indices_H.shape[0]}")
            continue

        # Kenar indekslerini kaydır (super-graph için)
        edg_indices_A_offset = edg_indices_A + [con_node_offset, var_node_offset]
        q_edg_indices_H_offset = q_edg_indices_H + [var_node_offset, var_node_offset]

        varFeatures_list.append(var_features)
        conFeatures_list.append(con_features)
        edgFeatures_A_list.append(edg_features_A)
        edgIndices_A_list.append(edg_indices_A_offset)
        q_edgFeatures_H_list.append(q_edg_features_H)
        q_edgIndices_H_list.append(q_edg_indices_H_offset)
        labels_list.append(labels_data)

        # Ofsetleri güncelle
        var_node_offset += var_features.shape[0]
        con_node_offset += con_features.shape[0]
    
    if not varFeatures_list:
        print(f"Uyarı: {folder_path} klasöründen hiç geçerli veri yüklenemedi. Program sonlandırılıyor.")
        exit(1)

    # Tüm listeleri birleştir
    varFeatures_all = np.vstack(varFeatures_list)
    conFeatures_all = np.vstack(conFeatures_list)
    edgFeatures_A_all = np.vstack(edgFeatures_A_list)
    edgIndices_A_all = np.vstack(edgIndices_A_list)
    q_edgFeatures_H_all = np.vstack(q_edgFeatures_H_list)
    q_edgIndices_H_all = np.vstack(q_edgIndices_H_list)
    labels_all = np.vstack(labels_list)

    # TensorFlow tensörlerine dönüştür
    varFeatures_tf = tf.constant(varFeatures_all, dtype=tf.float32)
    conFeatures_tf = tf.constant(conFeatures_all, dtype=tf.float32)
    edgFeatures_A_tf = tf.constant(edgFeatures_A_all, dtype=tf.float32)
    edgIndices_A_tf = tf.transpose(tf.constant(edgIndices_A_all, dtype=tf.int32))
    q_edgFeatures_H_tf = tf.constant(q_edgFeatures_H_all, dtype=tf.float32)
    q_edgIndices_H_tf = tf.transpose(tf.constant(q_edgIndices_H_all, dtype=tf.int32))
    labels_tf = tf.constant(labels_all, dtype=tf.float32)

    # Batch boyutları
    n_cons_total_batch = tf.constant(conFeatures_all.shape[0], dtype=tf.int32)
    n_vars_total_batch = tf.constant(varFeatures_all.shape[0], dtype=tf.int32)

    dataloader_tuple = (
        conFeatures_tf, edgIndices_A_tf, edgFeatures_A_tf,
        varFeatures_tf, q_edgIndices_H_tf, q_edgFeatures_H_tf,
        n_cons_total_batch, n_vars_total_batch,
        tf.constant(n_Cons_small, dtype=tf.int32),
        tf.constant(n_Vars_small, dtype=tf.int32),
        labels_tf
    )
    return dataloader_tuple

def process_train_step(model, dataloader, optimizer, type='fea'):
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)

    with tf.GradientTape() as tape:
        logits = model(batched_states, training=True)
        
        if type == "obj":
            loss = relative_loss(cand_scores, logits)
        elif type == "sol":
            # Solution için basit MSE loss kullan
            loss = solution_mse_loss(cand_scores, logits)
        else: # type == "fea"
            loss = tf.reduce_mean(tf.square(cand_scores - logits))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    err_rate = None
    if type == "fea":
        logits_sigmoid = tf.sigmoid(logits).numpy()
        cand_scores_np = cand_scores.numpy()
        
        errs_fp = np.sum((logits_sigmoid > 0.5) & (cand_scores_np < 0.5))
        errs_fn = np.sum((logits_sigmoid < 0.5) & (cand_scores_np > 0.5))
        total_errs = errs_fp + errs_fn
        err_rate = total_errs / cand_scores_np.shape[0]

    return loss.numpy(), err_rate

def process_eval(model, dataloader, type='fea'):
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)
    
    logits = model(batched_states, training=False)
    
    if type == "obj":
        loss = relative_loss(cand_scores, logits)
    elif type == "sol":
        # Solution için basit MSE loss kullan
        loss = solution_mse_loss(cand_scores, logits)
    else: # type == "fea"
        loss = tf.reduce_mean(tf.square(cand_scores - logits))

    err_rate = None
    if type == "fea":
        logits_sigmoid = tf.sigmoid(logits).numpy()
        cand_scores_np = cand_scores.numpy()
        errs_fp = np.sum((logits_sigmoid > 0.5) & (cand_scores_np < 0.5))
        errs_fn = np.sum((logits_sigmoid < 0.5) & (cand_scores_np > 0.5))
        total_errs = errs_fp + errs_fn
        err_rate = total_errs / cand_scores_np.shape[0]
        
    return loss.numpy(), err_rate

## SETUP HYPERPARAMETERS
max_epochs = args.epoch
lr = 0.0003
seed = 0
val_split = args.valSplit
weight_decay = args.weightDecay
dropout_rate = args.dropout
patience = args.patience

## DATASET SETUP
trainfolder = args.data_folder
valfolder = args.val_data_folder

# Gerçek veri boyutlarını otomatik algıla
sample_gnn_dir = os.path.join(args.data_folder, "Data_0", "Data_0")
try:
    sample_var = read_csv(os.path.join(sample_gnn_dir, "VarFeatures.csv"), header=None)
    sample_con = read_csv(os.path.join(sample_gnn_dir, "ConFeatures.csv"), header=None)
    sample_edge_A = read_csv(os.path.join(sample_gnn_dir, "EdgeIndices_A.csv"), header=None)
    sample_qedge_H = read_csv(os.path.join(sample_gnn_dir, "QEdgeIndices.csv"), header=None)
    
    n_Vars_small = sample_var.shape[0]
    n_Cons_small = sample_con.shape[0]
    n_Eles_small_A = sample_edge_A.shape[0]
    n_Eles_small_H = sample_qedge_H.shape[0]
    
    print(f'Gerçek veri boyutları (otomatik algılandı):')
    print(f'  Variables per graph: {n_Vars_small}')
    print(f'  Constraints per graph: {n_Cons_small}')
    print(f'  A edges per graph: {n_Eles_small_A}')
    print(f'  H edges per graph: {n_Eles_small_H}')
    
except FileNotFoundError as e:
    print(f"Uyarı: Otomatik boyut algılama başarısız: {e}")
    print("Manuel boyutlar kullanılıyor...")
    n_Vars_small = 10
    n_Cons_small = 40
    n_Eles_small_A = 400
    n_Eles_small_H = 100

## LOAD DATASET INTO MEMORY
train_dataloader = load_and_batch_data(
    args.data_folder, args.num_train_samples, n_Cons_small, n_Vars_small, 
    load_solution_labels=True, model_type=args.type
)
val_dataloader = load_and_batch_data(
    args.val_data_folder, args.num_test_samples, n_Cons_small, n_Vars_small, 
    load_solution_labels=True, model_type=args.type
)

if train_dataloader is None or val_dataloader is None:
    print("Hata: Veri yükleme başarısız oldu.")
    exit(1)

print('Train data shapes (after loading and batching):')
print(f'  Constraint features: {train_dataloader[0].shape}')
print(f'  Edge indices A: {train_dataloader[1].shape}')
print(f'  Edge features A: {train_dataloader[2].shape}')
print(f'  Variable features: {train_dataloader[3].shape}')
print(f'  QEdge indices H: {train_dataloader[4].shape}')
print(f'  QEdge features H: {train_dataloader[5].shape}')
print(f'  Total constraints in batch: {train_dataloader[6].numpy()}')
print(f'  Total variables in batch: {train_dataloader[7].numpy()}')
print(f'  Constraints per graph: {train_dataloader[8].numpy()}')
print(f'  Variables per graph: {train_dataloader[9].numpy()}')
print(f'  Labels: {train_dataloader[10].shape}')

print('\nValidation data shapes:')
print(f'  Constraint features: {val_dataloader[0].shape}')
print(f'  Edge indices A: {val_dataloader[1].shape}')
print(f'  Edge features A: {val_dataloader[2].shape}')
print(f'  Variable features: {val_dataloader[3].shape}')
print(f'  QEdge indices H: {val_dataloader[4].shape}')
print(f'  QEdge features H: {val_dataloader[5].shape}')
print(f'  Total constraints in batch: {val_dataloader[6].numpy()}')
print(f'  Total variables in batch: {val_dataloader[7].numpy()}')
print(f'  Constraints per graph: {val_dataloader[8].numpy()}')
print(f'  Variables per graph: {val_dataloader[9].numpy()}')
print(f'  Labels: {val_dataloader[10].shape}')

# Feature boyutlarını al
sample_dir = os.path.join(args.data_folder, "Data_0", "Data_0")
try:
    sample_var = read_csv(os.path.join(sample_dir, "VarFeatures.csv"), header=None)
    sample_con = read_csv(os.path.join(sample_dir, "ConFeatures.csv"), header=None)
    sample_edge_A = read_csv(os.path.join(sample_dir, "EdgeFeatures_A.csv"), header=None)
    sample_qedge_H = read_csv(os.path.join(sample_dir, "QEdgeFeatures.csv"), header=None)
    
    nVarF = sample_var.shape[1]
    nConsF = sample_con.shape[1] 
    nEdgeF = sample_edge_A.shape[1]
    nQEdgeF = sample_qedge_H.shape[1]
    
    print(f'\nFeature dimensions:')
    print(f'  Variable features: {nVarF}')
    print(f'  Constraint features: {nConsF}')
    print(f'  Edge features A: {nEdgeF}')
    print(f'  QEdge features H: {nQEdgeF}')
    
except FileNotFoundError:
    nVarF = 4
    nConsF = 4
    nEdgeF = 1
    nQEdgeF = 1

## SETUP MODEL
if not os.path.exists('./saved-models/'):
    os.makedirs('./saved-models/')
model_path = './saved-models/qp_' + args.type + '_s' + str(args.embSize) + '.pkl'

# Model parameters
output_units = 1
output_activation = None

if args.type == "fea":
    output_activation = 'sigmoid'
elif args.type == "sol":
    output_units = n_Vars_small
    output_activation = None
elif args.type == "obj":
    output_units = 1
    output_activation = None

## SETUP TENSORFLOW GPU
tf.random.set_seed(seed)
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    try:
        tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        print(f"GPU {gpu_index} yapılandırıldı: {gpus[gpu_index].name}")
    except RuntimeError as e:
        print(f"GPU ayarı atlandı (zaten yapılandırılmış): {e}")
else:
    print("GPU bulunamadı, CPU kullanılacak")

## MAIN TRAINING LOOP
with tf.device("GPU:" + str(gpu_index) if len(gpus) > 0 else "/CPU:0"):
    # Create model
    model = QPGNNPolicy(
        emb_size=args.embSize,
        cons_nfeats=nConsF,
        edge_nfeats=nEdgeF,
        var_nfeats=nVarF,
        qedge_nfeats=nQEdgeF,
        is_graph_level=(args.type != "sol"),
        output_units=output_units,
        output_activation=output_activation,
        dropout_rate=dropout_rate
    )
    
    # Optimizer
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        print("AdamW optimizer not available, using standard Adam optimizer")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Early stopping variables
    best_val_loss = float('inf')
    wait = 0
    best_epoch = 0

    print(f"\n🚀 Eğitim başlıyor...")
    print(f"Model tipi: {args.type}")
    print(f"Embedding boyutu: {args.embSize}")
    print(f"Maksimum epoch: {max_epochs}")
    print(f"Sabır (patience): {patience}")
    print("-" * 80)

    # Training loop
    for epoch in range(max_epochs):
        # Train step
        train_loss, train_err = process_train_step(model, train_dataloader, optimizer, type=args.type)
        
        # Validation step
        val_loss, val_err = process_eval(model, val_dataloader, type=args.type)
        
        # Print progress
        if args.type == "fea":
            print(f"Epoch {epoch:4d}: Train Loss={train_loss:.6f}, Train Err={train_err:.4f}, "
                  f"Val Loss={val_loss:.6f}, Val Err={val_err:.4f}")
        else:
            print(f"Epoch {epoch:4d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_epoch = epoch
            model.save_state(model_path)
            print(f"  ✓ Saved best model at epoch {epoch} with val_loss={val_loss:.6f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"  ✗ Early stopping after {patience} epochs without improvement")
                print(f"  ✓ Best model was at epoch {best_epoch} with val_loss={best_val_loss:.6f}")
                break

    print("-" * 80)
    print(f"🎉 Eğitim tamamlandı!")
    print(f"En iyi doğrulama kaybı: {best_val_loss:.6f} (epoch {best_epoch})")