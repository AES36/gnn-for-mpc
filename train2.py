import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
import random
from sklearn.model_selection import train_test_split
# QPGNNPolicy sınıfının models.py dosyasında tanımlı olduğunu varsayıyoruz.
# Eğer models.py dosyanız farklı bir konumdaysa veya sınıf adı farklıysa,
# bu satırı kendi projenize göre düzenlemeniz gerekebilir.
from models import QPGNNPolicy 

# --- Argümanların Tanımlanması ---
parser = argparse.ArgumentParser(description="QP GNN Model Eğitim Scripti")
parser.add_argument("--data_folder", help="Eğitim verilerinin bulunduğu klasör", default="./train_data", type=str)
parser.add_argument("--total_samples", help="Kullanılacak toplam eğitim örneği sayısı", default=2000, type=int)
parser.add_argument("--val_split", help="Doğrulama (validation) veri setinin oranı", default=0.2, type=float)
parser.add_argument("--gpu", help="Kullanılacak GPU indeksi (-1 CPU için)", default="0", type=str)
parser.add_argument("--emb_size", help="GNN gömme (embedding) boyutu", default=32, type=int)
parser.add_argument("--epochs", help="Maksimum epoch (eğitim turu) sayısı", default=100, type=int)
parser.add_argument("--type", help="Model türü", default="fea", choices=['fea','obj','sol']) # 'fea': olabilirlik, 'obj': amaç fonksiyonu, 'sol': çözüm
parser.add_argument("--lr_schedule", help="Öğrenme oranı (Learning Rate) çizelgesi türü", default="adaptive", choices=['fixed', 'adaptive', 'cosine', 'exponential'])
parser.add_argument("--dropout", help="Dropout oranı", default=0.1, type=float)
parser.add_argument("--weight_decay", help="Ağırlık düşüşü (L2 regularizasyon)", default=1e-5, type=float)
parser.add_argument("--verbose", help="Detaylı çıktı göster", action="store_true") # Bu bir flag argümanıdır, belirtilirse True olur
args = parser.parse_args()

# --- Adaptif Öğrenme Oranı Yöneticisi ---
class AdaptiveLRManager:
    """
    Model eğitiminde öğrenme oranını dinamik olarak yöneten sınıf.
    Farklı çizelgeler (sabit, adaptif, kosinüs, üstel) destekler.
    """
    def __init__(self, optimizer, schedule_type='adaptive', total_epochs=100):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        
        # Adaptif öğrenme oranı için parametreler
        self.patience = 10 # Doğrulama kaybının kaç epoch boyunca kötüleşmesine izin verileceği
        self.factor = 0.5  # Öğrenme oranının düşürüleceği çarpan
        self.min_lr = 1e-7 # Öğrenme oranının düşebileceği minimum değer
        self.wait = 0      # Doğrulama kaybının kötüleştiği ardışık epoch sayısı
        self.best_loss = float('inf') # Şimdiye kadarki en iyi doğrulama kaybı
        
        # Modelin türüne ve parametrelerine göre başlangıç öğrenme oranını otomatik belirler
        self.initial_lr = self._get_optimal_lr()
        self.current_lr = self.initial_lr
        optimizer.learning_rate.assign(self.initial_lr) # Optimizatörün öğrenme oranını ayarlar
        
        print(f"🎯 Otomatik seçilen başlangıç öğrenme oranı: {self.initial_lr:.2e} (çizelge: {schedule_type})")
        
    def _get_optimal_lr(self):
        """
        Deneyim ve araştırmalara dayalı olarak başlangıç öğrenme oranını otomatik olarak belirler.
        Modelin gömme boyutu ve tahmin türü (obje, çözüm, olabilirlik) dikkate alınır.
        """
        base_lr = 0.001 # Temel öğrenme oranı
        
        # Gömme boyutuna göre ayarlama: Daha büyük modeller genellikle daha küçük LR'lere ihtiyaç duyar
        emb_factor = min(1.0, 64 / max(args.emb_size, 1))
        
        # Model türüne göre ayarlama
        if hasattr(args, 'type'):
            if args.type == 'obj':
                base_lr *= 0.5  # Amaç fonksiyonu tahmini daha hassas öğrenme gerektirebilir
            elif args.type == 'sol':
                base_lr *= 0.8  # Çözüm tahmini orta karmaşıklıkta olabilir
            # 'fea' (olabilirlik) modeli için temel oran korunur
        
        return base_lr * emb_factor
        
    def _get_lr_for_epoch(self, epoch):
        """Belirli bir epoch için öğrenme oranını döndürür (adaptif olmayan çizelgeler için)."""
        if self.schedule_type == 'cosine':
            # Kosinüs soğutması (Cosine annealing): Öğrenme oranını kosinüs fonksiyonuna göre yavaşça düşürür
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))
        elif self.schedule_type == 'exponential':
            # Üstel düşüş (Exponential decay): Belirli aralıklarla öğrenme oranını düşürür
            decay_rate = 0.95
            return self.initial_lr * (decay_rate ** (epoch // 10)) # Her 10 epoch'ta bir düşüş
        else:
            return self.current_lr # Sabit öğrenme oranı için mevcut değeri döndür
    
    def step(self, epoch, val_loss=None):
        """
        Öğrenme oranını mevcut çizelge türüne göre günceller.
        Adaptif çizelge için doğrulama kaybını kullanır.
        """
        lr_changed = False # Öğrenme oranının bu adımda değişip değişmediğini izler
        
        if self.schedule_type == 'adaptive':
            if val_loss is not None:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.wait = 0 # En iyi kayıp bulundu, bekleme sayacını sıfırla
                else:
                    self.wait += 1 # Kayıp iyileşmedi, bekleme sayacını artır
                    
                if self.wait >= self.patience:
                    old_lr = self.current_lr
                    self.current_lr = max(old_lr * self.factor, self.min_lr) # Öğrenme oranını düşür
                    self.optimizer.learning_rate.assign(self.current_lr) # Optimizatörü güncelle
                    self.wait = 0 # Bekleme sayacını sıfırla
                    
                    if old_lr != self.current_lr:
                        print(f"  📉 Adaptif LR: {old_lr:.2e} → {self.current_lr:.2e}")
                        lr_changed = True
        
        elif self.schedule_type in ['cosine', 'exponential']:
            new_lr = self._get_lr_for_epoch(epoch)
            if abs(new_lr - self.current_lr) > 1e-8: # Önemli bir değişiklik varsa güncelle
                old_lr = self.current_lr
                self.current_lr = new_lr
                self.optimizer.learning_rate.assign(self.current_lr)
                print(f"  📉 Çizelgeli LR: {old_lr:.2e} → {self.current_lr:.2e}")
                lr_changed = True
        
        return lr_changed

# --- QP Model Eğitici Sınıfı ---
class QPModelTrainer:
    """
    Kare Programlama (QP) GNN modelinin eğitim sürecini baştan sona yönetir.
    GPU kurulumu, veri yükleme, model oluşturma, eğitim ve doğrulama adımlarını içerir.
    """
    def __init__(self):
        self.train_indices = None
        self.val_indices = None
        self.data_dimensions = {} # Yüklenen GNN verilerinin boyutlarını saklar
        self.model = None
        self.optimizer = None
        self.lr_manager = None
        self.obj_mean = None      # Amaç fonksiyonu normalizasyonu için ortalama
        self.obj_std = None       # Amaç fonksiyonu normalizasyonu için standart sapma
        
    def setup_gpu(self):
        """
        TensorFlow için GPU yapılandırmasını ayarlar. 
        Belirtilen GPU indeksini kullanır veya CPU'ya düşer.
        """
        if args.gpu == "-1":
            print("🖥️  CPU üzerinde çalışıyor.")
            tf.config.set_visible_devices([], 'GPU') # Tüm GPU'ları devre dışı bırak
            return "/CPU:0"
        else:
            gpu_index = int(args.gpu)
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 0:
                try:
                    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[gpu_index], True) # GPU bellek büyümesini etkinleştir
                    print(f"🚀 GPU {gpu_index} kullanılıyor: {gpus[gpu_index].name}")
                    return f"/GPU:{gpu_index}"
                except Exception as e:
                    print(f"⚠️  GPU kurulumu başarısız: {e}. CPU kullanılıyor.")
                    return "/CPU:0"
            else:
                print("❌ GPU bulunamadı. CPU kullanılıyor.")
                return "/CPU:0"
    
    def detect_data_dimensions(self):
        """
        Veri klasöründeki ilk örnek dosyayı okuyarak GNN girişlerinin boyutlarını algılar.
        Bu boyutlar modelin başlatılması için gereklidir.
        """
        sample_dir = os.path.join(args.data_folder, "Data_0", "Data_0") # İlk örnek veri dizini
        try:
            # Örnek dosyaları bir kez okuyarak boyutları çıkar
            sample_var = read_csv(os.path.join(sample_dir, "VarFeatures.csv"), header=None)
            sample_con = read_csv(os.path.join(sample_dir, "ConFeatures.csv"), header=None)
            sample_edge_A = read_csv(os.path.join(sample_dir, "EdgeFeatures_A.csv"), header=None)
            sample_qedge_H = read_csv(os.path.join(sample_dir, "QEdgeFeatures.csv"), header=None)
            
            self.data_dimensions = {
                'n_vars': sample_var.shape[0],          # Grafik başına değişken (düğüm) sayısı
                'n_cons': sample_con.shape[0],          # Grafik başına kısıt (düğüm) sayısı
                'var_features': sample_var.shape[1],    # Değişken düğüm özelliklerinin boyutu
                'con_features': sample_con.shape[1],    # Kısıt düğüm özelliklerinin boyutu
                'edge_features': sample_edge_A.shape[1], # Kenar özelliklerinin boyutu (Ax matrisleri için)
                'qedge_features': sample_qedge_H.shape[1] # Kare kenar özelliklerinin boyutu (Q matrisi için)
            }
            
            print(f"📊 Algılanan veri boyutları:")
            print(f"   Grafik başına değişken: {self.data_dimensions['n_vars']}")
            print(f"   Grafik başına kısıt: {self.data_dimensions['n_cons']}")
            print(f"   Değişken özellikleri: {self.data_dimensions['var_features']}")
            print(f"   Kısıt özellikleri: {self.data_dimensions['con_features']}")
            print(f"   Kenar özellikleri: {self.data_dimensions['edge_features']}")
            print(f"   QKenar özellikleri: {self.data_dimensions['qedge_features']}")
            
        except FileNotFoundError as e:
            print(f"❌ Boyut algılama başarısız: {e}. Varsayılan boyutlar kullanılıyor.")
            # Dosya bulunamazsa veya hata oluşursa varsayılan boyutları kullan
            self.data_dimensions = {
                'n_vars': 10, 'n_cons': 40, 'var_features': 3,
                'con_features': 2, 'edge_features': 1, 'qedge_features': 1
            }
    
    def create_train_val_split(self):
        """
        Mevcut veri örneklerinden eğitim ve doğrulama veri setlerini oluşturur.
        `'obj'` model türü için amaç fonksiyonu etiketlerinin Z-skor istatistiklerini hesaplar.
        """
        available_indices = []
        for i in range(args.total_samples):
            instance_dir = os.path.join(args.data_folder, f"Data_{i}")
            feas_path = os.path.join(instance_dir, "Labels_feas.csv")
            
            if not os.path.exists(feas_path):
                continue # Olabilirlik etiketi yoksa atla
                
            # 'obj' veya 'sol' modelleri için yalnızca mümkün (feasible) örnekleri dahil et
            if args.type in ["obj", "sol"]:
                try:
                    is_feasible = read_csv(feas_path, header=None).values[0,0]
                    if is_feasible == 0:
                        continue  # Olmayan durumları atla
                except:
                    continue # Okuma hatası olursa atla
            
            # Seçilen model türüne göre ilgili etiket dosyasının varlığını kontrol et
            if args.type == "obj":
                if not os.path.exists(os.path.join(instance_dir, "Labels_obj.csv")):
                    continue
            elif args.type == "sol":
                if not os.path.exists(os.path.join(instance_dir, "Labels_solu.csv")):
                    continue
                    
            available_indices.append(i) # Geçerli veri indeksini listeye ekle
        
        if len(available_indices) == 0:
            raise ValueError("Hata: Eğitim için geçerli veri bulunamadı!")
        
        # scikit-learn'den train_test_split kullanarak eğitim ve doğrulama indekslerini ayır
        self.train_indices, self.val_indices = train_test_split(
            available_indices,
            test_size=args.val_split, # Doğrulama veri setinin oranı
            random_state=42,          # Tekrarlanabilirlik için sabit rastgele durum
            shuffle=True              # Veriyi karıştır
        )

        # Eğer model türü 'obj' (amaç fonksiyonu tahmini) ise, normalizasyon için istatistikleri hesapla
        if args.type == "obj":
            print("Amaç fonksiyonu etiketleri için Z-skor normalizasyon istatistikleri hesaplanıyor...")
            all_obj_labels = []
            # Sadece eğitim verilerinden istatistikleri topla (veri sızıntısını önle)
            for i in self.train_indices: 
                instance_dir = os.path.join(args.data_folder, f"Data_{i}")
                try:
                    obj_label = read_csv(os.path.join(instance_dir, "Labels_obj.csv"), header=None).values[0,0]
                    all_obj_labels.append(obj_label)
                except Exception as e:
                    if args.verbose:
                        print(f"⚠️ İstatistik toplama sırasında Data_{i} için amaç etiketi yüklenemedi: {e}")
                    continue
            
            if all_obj_labels:
                self.obj_mean = np.mean(all_obj_labels) # Ortalamayı hesapla
                self.obj_std = np.std(all_obj_labels)   # Standart sapmayı hesapla
                # Sıfıra bölmeyi önlemek için, eğer standart sapma sıfırsa (tüm değerler aynıysa) 1.0 yap
                if self.obj_std == 0:
                    self.obj_std = 1.0 
                print(f"   Amaç Normalizasyonu İstatistikleri: Ortalama={self.obj_mean:.4f}, Std={self.obj_std:.4f}")
            else:
                print("   Uyarı: Normalizasyon için amaç etiketi bulunamadı. Normalizasyon atlanıyor.")
                self.obj_mean = 0.0 # Varsayılan olarak normalizasyon yapma
                self.obj_std = 1.0  # Varsayılan olarak normalizasyon yapma
        
        print(f"📊 Veri ayrımı:")
        print(f"   Toplam mevcut örnek: {len(available_indices)}")
        print(f"   Eğitim örnekleri: {len(self.train_indices)}")
        print(f"   Doğrulama örnekleri: {len(self.val_indices)}")
        print(f"   Ayırma oranı: {args.val_split:.1%}")
    
    def load_batch_data(self, indices):
        """
        Verilen indeksler için GNN giriş verilerini ve etiketlerini yükler ve 
        süper-grafik formatında birleştirir.
        """
        varFeatures_list = []
        conFeatures_list = []
        edgFeatures_A_list = []
        edgIndices_A_list = []
        q_edgFeatures_H_list = []
        q_edgIndices_H_list = []
        labels_list = []

        var_node_offset = 0 # Birleşik süper-grafikte değişken düğüm indekslerinin ofseti
        con_node_offset = 0 # Birleşik süper-grafikte kısıt düğüm indekslerinin ofseti

        for i in indices:
            instance_dir = os.path.join(args.data_folder, f"Data_{i}")
            gnn_data_dir = os.path.join(instance_dir, f"Data_{i}")
            
            try:
                # Etiketleri model türüne göre yükle
                if args.type == "fea":
                    feas_label = read_csv(os.path.join(instance_dir, "Labels_feas.csv"), header=None).values[0,0]
                    labels_data = np.array([[feas_label]])
                elif args.type == "obj":
                    labels_data = read_csv(os.path.join(instance_dir, "Labels_obj.csv"), header=None).values
                    # Amaç etiketlerine Z-skor normalizasyonu uygula
                    if self.obj_mean is not None and self.obj_std is not None:
                        labels_data = (labels_data - self.obj_mean) / self.obj_std
                elif args.type == "sol":
                    labels_data = read_csv(os.path.join(instance_dir, "Labels_solu.csv"), header=None).values

                # GNN özelliklerini CSV dosyalarından yükle
                var_features = read_csv(os.path.join(gnn_data_dir, "VarFeatures.csv"), header=None).values
                con_features = read_csv(os.path.join(gnn_data_dir, "ConFeatures.csv"), header=None).values
                edg_features_A = read_csv(os.path.join(gnn_data_dir, "EdgeFeatures_A.csv"), header=None).values
                edg_indices_A = read_csv(os.path.join(gnn_data_dir, "EdgeIndices_A.csv"), header=None).values
                q_edg_features_H = read_csv(os.path.join(gnn_data_dir, "QEdgeFeatures.csv"), header=None).values
                q_edg_indices_H = read_csv(os.path.join(gnn_data_dir, "QEdgeIndices.csv"), header=None).values

                # Birden fazla grafiği tek bir süper-grafikte birleştirmek için indeks ofsetlerini uygula
                # Kısıt-değişken kenarları için ofset
                edg_indices_A_offset = edg_indices_A + [con_node_offset, var_node_offset]
                # Değişken-değişken kenarları (kuadratik) için ofset
                q_edg_indices_H_offset = q_edg_indices_H + [var_node_offset, var_node_offset]

                # Tüm yüklenen verileri listelere ekle
                varFeatures_list.append(var_features)
                conFeatures_list.append(con_features)
                edgFeatures_A_list.append(edg_features_A)
                edgIndices_A_list.append(edg_indices_A_offset)
                q_edgFeatures_H_list.append(q_edg_features_H)
                q_edgIndices_H_list.append(q_edg_indices_H_offset)
                labels_list.append(labels_data)

                # Sonraki grafik için ofsetleri güncelle
                var_node_offset += var_features.shape[0]
                con_node_offset += con_features.shape[0]
                
            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Data_{i} yüklenemedi: {e}. Bu örnek atlandı.")
                continue # Hata durumunda bu örneği atla

        # Tüm listelenen verileri tek NumPy dizilerinde birleştir
        varFeatures_all = np.vstack(varFeatures_list)
        conFeatures_all = np.vstack(conFeatures_list)
        edgFeatures_A_all = np.vstack(edgFeatures_A_list)
        edgIndices_A_all = np.vstack(edgIndices_A_list)
        q_edgFeatures_H_all = np.vstack(q_edgFeatures_H_list)
        q_edgIndices_H_all = np.vstack(q_edgIndices_H_list)
        labels_all = np.vstack(labels_list)

        # NumPy dizilerini TensorFlow tensörlerine dönüştür ve batch verisini oluştur
        batch_data = (
            tf.constant(conFeatures_all, dtype=tf.float32),
            tf.transpose(tf.constant(edgIndices_A_all, dtype=tf.int32)), # Kenar indeksleri için transpozisyon gerekli olabilir
            tf.constant(edgFeatures_A_all, dtype=tf.float32),
            tf.constant(varFeatures_all, dtype=tf.float32),
            tf.transpose(tf.constant(q_edgIndices_H_all, dtype=tf.int32)), # Kare kenar indeksleri için transpozisyon gerekli olabilir
            tf.constant(q_edgFeatures_H_all, dtype=tf.float32),
            tf.constant(conFeatures_all.shape[0], dtype=tf.int32), # Toplam kısıt düğümü sayısı
            tf.constant(varFeatures_all.shape[0], dtype=tf.int32), # Toplam değişken düğümü sayısı
            tf.constant(self.data_dimensions['n_cons'], dtype=tf.int32), # Her bir grafikteki kısıt sayısı (GNN katmanları için gerekli)
            tf.constant(self.data_dimensions['n_vars'], dtype=tf.int32), # Her bir grafikteki değişken sayısı (GNN katmanları için gerekli)
            tf.constant(labels_all, dtype=tf.float32) # Eğitim etiketleri
        )
        
        return batch_data
    
    def create_model(self):
        """
        GNN modelini (QPGNNPolicy) belirtilen argümanlara göre oluşturur ve 
        TensorFlow optimizatörünü (AdamW) başlatır.
        """
        output_units = 1
        output_activation = None # Varsayılan olarak aktivasyon yok (lineer çıktı)
        
        if args.type == "fea":
            output_activation = 'sigmoid' # Olabilirlik için 0-1 aralığında çıktı
        elif args.type == "sol":
            output_units = self.data_dimensions['n_vars'] # Çözüm vektörü için değişken sayısı kadar çıktı
            output_activation = None
        elif args.type == "obj":
            output_units = 1 # Amaç fonksiyonu için tek bir skaler çıktı
            output_activation = None # Z-skor normalizasyonu sonrası genellikle lineer çıktı kullanılır
        
        self.model = QPGNNPolicy(
            emb_size=args.emb_size,
            cons_nfeats=self.data_dimensions['con_features'],
            edge_nfeats=self.data_dimensions['edge_features'],
            var_nfeats=self.data_dimensions['var_features'],
            qedge_nfeats=self.data_dimensions['qedge_features'],
            is_graph_level=(args.type != "sol"), # 'sol' tipi düğüm seviyesinde, diğerleri grafik seviyesinde tahmin yapar
            output_units=output_units,
            output_activation=output_activation,
            dropout_rate=args.dropout
        )
        
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,  # Öğrenme oranı, LR yöneticisi tarafından üzerine yazılacaktır
            weight_decay=args.weight_decay
        )
        
        self.lr_manager = AdaptiveLRManager(
            self.optimizer,
            schedule_type=args.lr_schedule,
            total_epochs=args.epochs
        )
        
        print(f"🔧 Model başarıyla oluşturuldu:")
        print(f"   Tipi: {args.type}")
        print(f"   Gömme boyutu: {args.emb_size}")
        print(f"   Çıkış birimleri: {output_units}")
        print(f"   Dropout oranı: {args.dropout}")
        print(f"   LR Çizelgesi: {args.lr_schedule}")
        print(f"   Ağırlık düşüşü: {args.weight_decay}")
    
    def compute_loss(self, y_true, y_pred):
        """
        Model türüne göre uygun kayıp fonksiyonunu hesaplar.
        'obj' için bağıl mutlak hata, 'sol' ve 'fea' için ortalama kare hata kullanılır.
        """
        if args.type == "obj":
            # Amaç fonksiyonu için bağıl mutlak hata (relative absolute error)
            # Normalize edilmiş değerler için de etkili olabilir.
            epsilon = 1e-6 # Sıfıra bölmeyi önlemek için küçük bir değer
            return tf.reduce_mean(tf.abs(y_true - y_pred) / (tf.abs(y_true) + epsilon))
            # Alternatif olarak, normalize edilmiş değerler için doğrudan MSE de iyi çalışabilir:
            # return tf.reduce_mean(tf.square(y_true - y_pred))
        elif args.type == "sol":
            # Çözüm tahmini için Ortalama Kare Hata (Mean Squared Error)
            return tf.reduce_mean(tf.square(y_true - y_pred))
        else:  # args.type == "fea" (Olabilirlik)
            # Olabilirlik tahmini için Ortalama Kare Hata (Mean Squared Error)
            return tf.reduce_mean(tf.square(y_true - y_pred))
    
    @tf.function # TensorFlow grafiği olarak derlenerek performans artışı sağlar
    def train_step(self, batch_data):
        """Tek bir eğitim adımını gerçekleştirir (ileri yayılım, kayıp hesaplama, geri yayılım, ağırlık güncelleme)."""
        *batched_states, labels = batch_data # Batch verilerini ve etiketleri ayır
        
        with tf.GradientTape() as tape: # Gradyanları kaydetmek için GradientTape kullan
            predictions = self.model(batched_states, training=True) # Modeli eğitim modunda çalıştır
            loss = self.compute_loss(labels, predictions) # Tahminler ve gerçek etiketler arasındaki kaybı hesapla
        
        gradients = tape.gradient(loss, self.model.trainable_variables) # Modelin eğitilebilir değişkenleri için gradyanları hesapla
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # Gradyanları uygulayarak model ağırlıklarını güncelle
        
        return loss # Hesaplanan kaybı döndür
    
    @tf.function # TensorFlow grafiği olarak derlenerek performans artışı sağlar
    def val_step(self, batch_data):
        """Tek bir doğrulama adımını gerçekleştirir (ileri yayılım, kayıp hesaplama)."""
        *batched_states, labels = batch_data
        
        predictions = self.model(batched_states, training=False) # Modeli çıkarım (değerlendirme) modunda çalıştır
        loss = self.compute_loss(labels, predictions) # Kaybı hesapla
        
        return loss # Hesaplanan kaybı döndür
    
    def train(self):
        """Modelin ana eğitim döngüsünü yönetir."""
        print("📂 Eğitim verileri yükleniyor...")
        train_data = self.load_batch_data(self.train_indices)
        
        print("📂 Doğrulama verileri yükleniyor...")
        val_data = self.load_batch_data(self.val_indices)
        
        best_val_loss = float('inf') # Şimdiye kadarki en iyi doğrulama kaybını tutar
        best_epoch = 0              # En iyi kaybın elde edildiği epoch'u tutar
        model_path = f'./saved-models/qp_{args.type}_s{args.emb_size}.pkl' # Modelin kaydedileceği dosya yolu
        
        print(f"\n🚀 Eğitim başladı!")
        print(f"Model: {args.type} | Epochlar: {args.epochs} | Erken Durma Yok (Best Model Saved)")
        print("-" * 80)
        
        for epoch in range(args.epochs):
            # Eğitim ve doğrulama adımlarını çalıştır
            train_loss = self.train_step(train_data).numpy() # TensorFlow tensörünü NumPy değerine dönüştür
            val_loss = self.val_step(val_data).numpy()       # TensorFlow tensörünü NumPy değerine dönüştür
            
            # Adaptif öğrenme oranı yöneticisini güncelle
            lr_changed = self.lr_manager.step(epoch, val_loss)
            current_lr = self.optimizer.learning_rate.numpy()
            
            # Eğitim ilerlemesini konsola yazdır
            # Amaç fonksiyonu için doğrulama kaybının normalleştirilmiş olduğunu belirtiyoruz
            print(f"Epoch {epoch:4d}: Eğitim Kaybı={train_loss:.6f}, Doğrulama Kaybı (Normalize Edilmiş)={val_loss:.6f}, LR={current_lr:.2e}")
            
            # Eğer mevcut doğrulama kaybı şimdiye kadarki en iyiyse modeli kaydet
            # Not: Erken durma (early stopping) uygulanmadığı için eğitim tüm epoch'ları tamamlayacak,
            # ancak en iyi performans gösteren model kaydedilecektir.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                self.model.save_state(model_path) # Modelin ağırlıklarını ve durumunu kaydeder
                print(f"  ✓ En iyi model kaydedildi (doğrulama_kaybı={val_loss:.6f})")
        
        print("-" * 80)
        print(f"🎉 Eğitim tamamlandı!")
        print(f"En iyi doğrulama kaybı (normalize edilmiş): {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"Final model kaydedildi: {model_path}")
    
    def run(self):
        """Tüm eğitim sürecini başlatan ana fonksiyondur."""
        print("🚀 QP GNN Model Eğitimi Başlatılıyor...")
        print("=" * 60)
        
        # GPU'yu ayarla veya CPU'ya düş
        device = self.setup_gpu()
        # Veri boyutlarını algıla (modelin başlatılması için gerekli)
        self.detect_data_dimensions()
        # Eğitim ve doğrulama ayrımını oluştur ve 'obj' için normalizasyon istatistiklerini hesapla
        self.create_train_val_split()
        
        # Belirtilen cihazda (GPU veya CPU) model oluşturma ve eğitimi başlat
        with tf.device(device):
            self.create_model()
            
            # Modelleri kaydetmek için dizin oluştur
            os.makedirs('./saved-models', exist_ok=True)
            
            # Eğitimi başlat
            self.train()

if __name__ == "__main__":
    trainer = QPModelTrainer()
    trainer.run()