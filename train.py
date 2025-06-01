import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
# QPGNNPolicy modelinizi içeren dosya adını doğru bir şekilde buraya girin.
# Eğer models.py dosyanızın adı 'models.py' ise, sadece 'models' olarak içe aktarın.
# Eğer 'qp_models.py' ise, 'qp_models' olarak içe aktarın.
from models import QPGNNPolicy # Varsayılan olarak models.py'yi kullanıyorum

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser(description="QP GNN Modelini Eğitme Betiği.")
parser.add_argument("--data_folder", help="Eğitim veri setinin ana klasörü (örn: ./train_data)", default="./train_data", type=str)
parser.add_argument("--val_data_folder", help="Doğrulama veri setinin ana klasörü (örn: ./test_data)", default="./test_data", type=str) # Test verisi artık doğrulama için kullanılacak
parser.add_argument("--total_samples", help="Toplam eğitim + doğrulama örnek sayısı (veri klasöründe kaç adet Data_X var).", default=100, type=int) # args.data yerine total_samples oldu
parser.add_argument("--gpu", help="Kullanılacak GPU indeksi", default="0", type=str)
parser.add_argument("--embSize", help="GNN'nin gömme boyutu", default=64, type=int)
parser.add_argument("--epoch", help="Maksimum epoch sayısı", default=500, type=int)
parser.add_argument("--type", help="Model tipi: 'fea' (fizibilite), 'obj' (amaç değeri), 'sol' (çözüm)", default="fea", choices=['fea','obj','sol'])
parser.add_argument("--valSplit", help="Doğrulama için verinin oranı (0~1). Not: Şimdi ayrı klasörler kullanıldığı için bu argüman daha az doğrudan etkilidir, ancak tutarlılık için tutulabilir.", default=0.2, type=float)
parser.add_argument("--dropout", help="Dropout oranı", default=0.0, type=float)
parser.add_argument("--weightDecay", help="AdamW için ağırlık azaltma oranı", default=0.0, type=float)
parser.add_argument("--patience", help="Erken durdurma sabrı (kaç epoch iyileşme olmazsa durulacak)", default=300, type=int)
args = parser.parse_args()

## HELPER FUNCTIONS
def relative_loss(y_true, y_pred):
    """
    Göreceli mutlak hata: |y_true - y_pred| / (|y_true| + epsilon)
    Büyük objektif değerleri için daha dengeli bir eğitim sağlar.
    """
    epsilon = 1e-6 # Sıfıra bölmeyi önlemek için daha küçük bir epsilon değeri
    return tf.reduce_mean(tf.abs(y_true - y_pred) / (tf.abs(y_true) + epsilon))

def normalized_euclidean_loss(y_true, y_pred, n_Vars_small):
    """
    Normalize edilmiş Euclidean mesafe loss fonksiyonu.
    Çözüm vektörlerinin tahmininde daha anlamlı bir metrik.
    """
    # y_true ve y_pred batch_size * n_Vars_small, 1 boyutunda gelir
    # Her bir örnek için ayrı ayrı yeniden şekillendirip norm hesaplamalıyız
    batch_size = tf.shape(y_true)[0] // n_Vars_small
    y_true_reshaped = tf.reshape(y_true, [batch_size, n_Vars_small])
    y_pred_reshaped = tf.reshape(y_pred, [batch_size, n_Vars_small])
    
    distances = tf.norm(y_true_reshaped - y_pred_reshaped, axis=1) # tf.norm Euclidean norm için
    norms = tf.norm(y_true_reshaped, axis=1) + 1e-6 # Sıfıra bölmeyi önlemek için küçük epsilon
    return tf.reduce_mean(distances / norms)

# Veri yükleme fonksiyonu
def load_and_batch_data(folder_path, n_samples, n_Cons_small, n_Vars_small, load_solution_labels=True, model_type='fea'):
    """
    Veri klasöründen graf verilerini yükler ve GNN modelinin beklediği formata hazırlar.
    Bu, bir minibatch için tüm graf örneklerini birleştiren bir "super-graph" oluşturur.
    """
    varFeatures_list = []
    conFeatures_list = []
    edgFeatures_A_list = []
    edgIndices_A_list = []
    q_edgFeatures_H_list = []
    q_edgIndices_H_list = []
    labels_list = []

    # Her bir graf örneği için düğüm ve kenar indeks ofsetlerini takip etmek için
    var_node_offset = 0
    con_node_offset = 0

    for i in range(n_samples):
        instance_dir = os.path.join(folder_path, f"Data_{i}")
        
        # Fizibilite etiketini kontrol et, yoksa veya 0 ise atla
        feas_path = os.path.join(instance_dir, "Labels_feas.csv")
        if not os.path.exists(feas_path):
            print(f"Uyarı: {feas_path} bulunamadı, örnek {i} atlanıyor.")
            continue
        
        is_feasible = read_csv(feas_path, header=None).values[0,0]

        # Eğer model 'obj' veya 'sol' ise ve problem fizibil değilse atla
        if model_type in ["obj", "sol"] and is_feasible == 0:
            continue
        
        # Fizibilite modeli ise, Labels_feas her zaman okunur
        # Obj veya Sol modeli ise, sadece fizibil olanlar okunur
        if model_type == "fea":
            labels_data = np.array([[is_feasible]]) # Labels_feas.csv'den oku
        elif model_type == "obj":
            obj_path = os.path.join(instance_dir, "Labels_obj.csv")
            if not os.path.exists(obj_path): # Eğer fizibil değilse obj dosyası olmazdı
                continue 
            labels_data = read_csv(obj_path, header=None).values
        elif model_type == "sol":
            solu_path = os.path.join(instance_dir, "Labels_solu.csv")
            if not os.path.exists(solu_path): # Eğer fizibil değilse solu dosyası olmazdı
                continue
            labels_data = read_csv(solu_path, header=None).values


        # Verileri oku
        try:
            var_features = read_csv(os.path.join(instance_dir, "VarFeatures.csv"), header=None).values
            con_features = read_csv(os.path.join(instance_dir, "ConFeatures.csv"), header=None).values
            edg_features_A = read_csv(os.path.join(instance_dir, "EdgeFeatures_A.csv"), header=None).values
            edg_indices_A = read_csv(os.path.join(instance_dir, "EdgeIndices_A.csv"), header=None).values
            q_edg_features_H = read_csv(os.path.join(instance_dir, "QEdgeFeatures.csv"), header=None).values
            q_edg_indices_H = read_csv(os.path.join(instance_dir, "QEdgeIndices.csv"), header=None).values
        except FileNotFoundError as e:
            print(f"Hata: {e}. Örnek {i} için dosya bulunamadı, atlanıyor.")
            continue
        
        # Boyut kontrolleri (Super-graph oluşturmak için önemli)
        if (var_features.shape[0] != n_Vars_small or 
            con_features.shape[0] != n_Cons_small or 
            edg_indices_A.shape[0] != n_Cons_small * n_Vars_small or # m*n kenar
            q_edg_indices_H.shape[0] != n_Vars_small * n_Vars_small): # n*n kenar
            print(f"Uyarı: Örnek {i} için beklenen boyutlar eşleşmiyor, atlanıyor.")
            print(f"Beklenen: var_feats={n_Vars_small}, con_feats={n_Cons_small}, A_edges={n_Cons_small*n_Vars_small}, H_edges={n_Vars_small*n_Vars_small}")
            print(f"Bulunan: var_feats={var_features.shape[0]}, con_feats={con_features.shape[0]}, A_edges={edg_indices_A.shape[0]}, H_edges={q_edg_indices_H.shape[0]}")
            continue


        # Kenar indekslerini her örnek için kaydır (super-graph oluşturmak için)
        edg_indices_A_offset = edg_indices_A + [con_node_offset, var_node_offset]
        q_edg_indices_H_offset = q_edg_indices_H + [var_node_offset, var_node_offset] # Q edges sadece var'dan var'a


        varFeatures_list.append(var_features)
        conFeatures_list.append(con_features)
        edgFeatures_A_list.append(edg_features_A)
        edgIndices_A_list.append(edg_indices_A_offset)
        q_edgFeatures_H_list.append(q_edg_features_H)
        q_edgIndices_H_list.append(q_edg_indices_H_offset)
        labels_list.append(labels_data)

        # Ofsetleri güncelle
        var_node_offset += var_features.shape[0] # Her bir grafikteki değişken sayısı
        con_node_offset += con_features.shape[0] # Her bir grafikteki kısıt sayısı
    
    if not varFeatures_list: # Hiç veri yüklenemedi mi kontrol et
        print(f"Uyarı: {folder_path} klasöründen hiç geçerli veri yüklenemedi. Program sonlandırılıyor.")
        exit(1)


    # Tüm listeleri tek NumPy dizilerine birleştir
    varFeatures_all = np.vstack(varFeatures_list)
    conFeatures_all = np.vstack(conFeatures_list)
    edgFeatures_A_all = np.vstack(edgFeatures_A_list)
    edgIndices_A_all = np.vstack(edgIndices_A_list)
    q_edgFeatures_H_all = np.vstack(q_edgFeatures_H_list)
    q_edgIndices_H_all = np.vstack(q_edgIndices_H_list)
    labels_all = np.vstack(labels_list) # label'lar zaten tek değer veya vektör olarak gelir

    # TensorFlow tensörlerine dönüştür
    varFeatures_tf = tf.constant(varFeatures_all, dtype=tf.float32)
    conFeatures_tf = tf.constant(conFeatures_all, dtype=tf.float32)
    edgFeatures_A_tf = tf.constant(edgFeatures_A_all, dtype=tf.float32)
    edgIndices_A_tf = tf.transpose(tf.constant(edgIndices_A_all, dtype=tf.int32)) # Transpoze GNN formatı için
    q_edgFeatures_H_tf = tf.constant(q_edgFeatures_H_all, dtype=tf.float32)
    q_edgIndices_H_tf = tf.transpose(tf.constant(q_edgIndices_H_all, dtype=tf.int32)) # Transpoze GNN formatı için
    labels_tf = tf.constant(labels_all, dtype=tf.float32)

    # Toplu graf boyutları
    n_cons_total_batch = tf.constant(conFeatures_all.shape[0], dtype=tf.int32)
    n_vars_total_batch = tf.constant(varFeatures_all.shape[0], dtype=tf.int32)

    # Modülün beklediği demet formatı
    dataloader_tuple = (
        conFeatures_tf, edgIndices_A_tf, edgFeatures_A_tf,
        varFeatures_tf, q_edgIndices_H_tf, q_edgFeatures_H_tf,
        n_cons_total_batch, n_vars_total_batch,
        tf.constant(n_Cons_small, dtype=tf.int32), # Her bir grafikteki kısıt sayısı
        tf.constant(n_Vars_small, dtype=tf.int32), # Her bir grafikteki değişken sayısı
        labels_tf
    )
    return dataloader_tuple


def process_train_step(model, dataloader, optimizer, type='fea'):
    """Tek bir eğitim adımını işler."""
    # Dataloader'dan verileri aç
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)

    with tf.GradientTape() as tape:
        # training=True => Dropout aktif
        logits = model(batched_states, training=True)
        
        # Farklı model tipleri için farklı loss fonksiyonları
        if type == "obj":
            loss = relative_loss(cand_scores, logits)
        elif type == "sol":
            # n_vsm (n_Vars_small) argüman olarak geçirilmelidir
            loss = normalized_euclidean_loss(cand_scores, logits, n_Vars_small=n_vsm) 
        else: # type == "fea"
            # Fizibilite için genellikle Binary Crossentropy kullanılır, ancak orijinal kod MSE kullanmış.
            # Binary Crossentropy için logits sigmoid aktivasyonlu olmalı.
            # Eğer model çıktısı sigmoid ise tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # Eğer model çıktısı raw logits ise tf.keras.losses.BinaryCrossentropy(from_logits=True)
            # Şu anki modelde `output_activation=None` ve `output_units=1` olduğu için MSE'ye devam edelim.
            # Eğer sigmoid eklenecekse BinaryCrossentropy daha uygun olur.
            loss_tensor = tf.keras.losses.mean_squared_error(cand_scores, logits)
            loss = tf.reduce_mean(loss_tensor)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    err_rate = None
    if type == "fea":
        # Sınıflandırma hatasını hesapla (accuracy)
        # Logits'i (0,1) aralığına sıkıştırıp 0.5 eşiğiyle sınıflandırma yap
        logits_sigmoid = tf.sigmoid(logits).numpy() # Eğer modelin son katmanında sigmoid yoksa
        cand_scores_np = cand_scores.numpy()
        
        # Hataları hesapla: Yanlış pozitifler ve yanlış negatifler
        errs_fp = np.sum((logits_sigmoid > 0.5) & (cand_scores_np < 0.5))
        errs_fn = np.sum((logits_sigmoid < 0.5) & (cand_scores_np > 0.5))
        total_errs = errs_fp + errs_fn
        err_rate = total_errs / cand_scores_np.shape[0]

    return loss.numpy(), err_rate

def process_eval(model, dataloader, type='fea'):
    """Tek bir değerlendirme adımını işler."""
    c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm, cand_scores = dataloader
    batched_states = (c, ei, ev, v, qi, qv, n_cs, n_vs, n_csm, n_vsm)
    
    # training=False => Dropout inaktif
    logits = model(batched_states, training=False)
    
    # Farklı model tipleri için farklı loss fonksiyonları
    if type == "obj":
        loss = relative_loss(cand_scores, logits)
    elif type == "sol":
        loss = normalized_euclidean_loss(cand_scores, logits, n_Vars_small=n_vsm)
    else: # type == "fea"
        loss_tensor = tf.keras.losses.mean_squared_error(cand_scores, logits)
        loss = tf.reduce_mean(loss_tensor)

    err_rate = None
    if type == "fea":
        # Sınıflandırma hatasını hesapla
        logits_sigmoid = tf.sigmoid(logits).numpy() # Eğer modelin son katmanında sigmoid yoksa
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
val_split = args.valSplit # Artık doğrudan klasör ayrımı olduğu için bu argüman daha az kritik
weight_decay = args.weightDecay
dropout_rate = args.dropout
patience = args.patience

## DATASET SETUP
trainfolder = args.data_folder # './train_data'
valfolder = args.val_data_folder # './test_data'

# Sabit olarak her bir grafikteki kısıt ve değişken sayıları
# Bunlar veri üretiminizdeki N ve nx, nu değerlerinden türetilmelidir.
# N = 10, nx = 2, nu = 1 varsayımıyla:
n_Cons_small = 2 * args.N * args.nx + 2 * args.N * args.nu # 2*10*2 + 2*10*1 = 40 + 20 = 60
n_Vars_small = args.N * args.nu # 10*1 = 10

# Kenar sayıları dinamik olarak okunur, ancak ortalama bir değer belirtmek iyidir.
# Yoğun graf temsili kullandığımız için:
n_Eles_small_A = n_Cons_small * n_Vars_small # 60 * 10 = 600
n_Eles_small_H = n_Vars_small * n_Vars_small # 10 * 10 = 100

## LOAD DATASET INTO MEMORY
# load_and_batch_data fonksiyonu, tüm veriyi memory'ye yükler
# Büyük veri setleri için tf.data.Dataset kullanılmalı, ancak küçük/orta için bu uygun
train_dataloader = load_and_batch_data(
    args.data_folder, args.num_train_samples, n_Cons_small, n_Vars_small, 
    load_solution_labels=True, model_type=args.type
)
val_dataloader = load_and_batch_data(
    args.val_data_folder, args.num_test_samples, n_Cons_small, n_Vars_small, 
    load_solution_labels=True, model_type=args.type
)

# Debug: Veri yükleme sonrası boyutları kontrol et
if train_dataloader is None or val_dataloader is None:
    print("Hata: Veri yükleme başarısız oldu. Lütfen klasör yollarını ve örnek sayılarını kontrol edin.")
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

print('\nValidation data shapes (after loading and batching):')
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


## SETUP MODEL AND SAVED MODEL PATH
if not os.path.exists('./saved-models/'):
    os.makedirs('./saved-models/')
model_path = './saved-models/qp_' + args.type + '_s' + str(args.embSize) + '.pkl'

# Model output units and activation based on type
output_units = 1
output_activation = None # default for obj and sol

if args.type == "fea":
    output_activation = 'sigmoid' # Feasibility is binary classification
elif args.type == "sol":
    output_units = n_Vars_small # Solution is a vector of size n_Vars_small
    output_activation = None # Solution is real-valued, no activation
elif args.type == "obj":
    output_units = 1 # Objective is a single real value
    output_activation = None # Objective is real-valued, no activation


## SETUP TENSORFLOW GPU
tf.random.set_seed(seed)
gpu_index = int(args.gpu)
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

## MAIN TRAINING LOOP
with tf.device("GPU:" + str(gpu_index) if len(gpus) > 0 else "/CPU:0"):
    # Create model instance
    model = QPGNNPolicy(
        emb_size=args.embSize,
        cons_nfeats=nConsF,
        edge_nfeats=nEdgeF,
        var_nfeats=nVarF,
        qedge_nfeats=nQEdgeF,
        is_graph_level=(args.type != "sol"), # isGraphLevel True for fea/obj, False for sol
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

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    wait = 0
    best_epoch = 0

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

    print(f"Training completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")