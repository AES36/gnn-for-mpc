# single_inference.py - Tek veri için model kullanımı
import tensorflow as tf
import numpy as np
from models import QPGNNPolicy
from pandas import read_csv
import argparse

parser = argparse.ArgumentParser(description="Tek veri için GNN model inference")
parser.add_argument("--data_folder", default="./test_data", help="Veri klasörü")
parser.add_argument("--data_id", type=int, default=0, help="Data_X numarası")
parser.add_argument("--model_path", default="./saved_models/qp_sol_emb32_N10nx2nu1/ckpt-1", help="Model checkpoint yolu")
args = parser.parse_args()

def load_trained_model(checkpoint_path):
    """Eğitilmiş modeli yükle"""
    model = QPGNNPolicy(
        emb_size=32,
        cons_nfeats=2,
        edge_nfeats=1,
        var_nfeats=3,
        qedge_nfeats=1,
        is_graph_level=False,  # Solution task
        output_units=10,       # N*nu = 10*1
        output_activation=None,
        dropout_rate=0.0
    )
    
    # Dummy input ile model build et
    dummy_input = (
        tf.zeros((40, 2)),
        tf.zeros((2, 400), dtype=tf.int32),
        tf.zeros((400, 1)),
        tf.zeros((10, 3)),
        tf.zeros((2, 100), dtype=tf.int32),
        tf.zeros((100, 1)),
        tf.constant(40, dtype=tf.int32),
        tf.constant(10, dtype=tf.int32),
        tf.constant(40, dtype=tf.int32),
        tf.constant(10, dtype=tf.int32)
    )
    _ = model(dummy_input, training=False)
    
    # Checkpoint yükle
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()
    
    return model

def prepare_single_data(data_folder, data_id):
    """Tek bir Data_X örneğini model input formatına çevir"""
    instance_dir = f"{data_folder}/Data_{data_id}"
    
    try:
        # Graph features yükle
        var_features = read_csv(f"{instance_dir}/VarFeatures.csv", header=None).values
        con_features = read_csv(f"{instance_dir}/ConFeatures.csv", header=None).values
        edge_features_A = read_csv(f"{instance_dir}/EdgeFeatures_A.csv", header=None).values
        edge_indices_A = read_csv(f"{instance_dir}/EdgeIndices_A.csv", header=None).values
        qedge_features_H = read_csv(f"{instance_dir}/QEdgeFeatures.csv", header=None).values
        qedge_indices_H = read_csv(f"{instance_dir}/QEdgeIndices.csv", header=None).values
        
        # Model input tuple
        model_input = (
            tf.constant(con_features, dtype=tf.float32),
            tf.transpose(tf.constant(edge_indices_A, dtype=tf.int32)),
            tf.constant(edge_features_A, dtype=tf.float32),
            tf.constant(var_features, dtype=tf.float32),
            tf.transpose(tf.constant(qedge_indices_H, dtype=tf.int32)),
            tf.constant(qedge_features_H, dtype=tf.float32),
            tf.constant(con_features.shape[0], dtype=tf.int32),
            tf.constant(var_features.shape[0], dtype=tf.int32),
            tf.constant(con_features.shape[0], dtype=tf.int32),
            tf.constant(var_features.shape[0], dtype=tf.int32)
        )
        
        return model_input
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        return None

def get_true_solution(data_folder, data_id):
    """Gerçek çözümü al (varsa)"""
    try:
        true_sol = read_csv(f"{data_folder}/Data_{data_id}/Labels_solu.csv", header=None).values.flatten()
        return true_sol
    except:
        return None

def main():
    print("🚀 Tek Veri için GNN Model Inference")
    print("="*50)
    
    # Model yükle
    print(f"📦 Model yükleniyor: {args.model_path}")
    model = load_trained_model(args.model_path)
    print("✅ Model başarıyla yüklendi!")
    
    # Veri hazırla
    print(f"📂 Veri hazırlanıyor: {args.data_folder}/Data_{args.data_id}")
    single_data = prepare_single_data(args.data_folder, args.data_id)
    
    if single_data is None:
        print("❌ Veri hazırlanamadı!")
        return
    
    print("✅ Veri başarıyla hazırlandı!")
    
    # Inference
    print("\n🧠 Model inference yapılıyor...")
    import time
    start_time = time.time()
    
    prediction = model(single_data, training=False)
    control_solution = prediction.numpy().flatten()
    
    inference_time = time.time() - start_time
    
    # Sonuçları göster
    print(f"\n📊 Sonuçlar:")
    print(f"   Inference Time: {inference_time:.4f} saniye")
    print(f"   Control Solution Shape: {control_solution.shape}")
    print(f"   Control Values: {control_solution}")
    print(f"   Value Range: [{control_solution.min():.4f}, {control_solution.max():.4f}]")
    
    # Gerçek çözümle karşılaştır
    true_solution = get_true_solution(args.data_folder, args.data_id)
    if true_solution is not None:
        error = np.mean(np.abs(control_solution - true_solution))
        relative_error = error / (np.mean(np.abs(true_solution)) + 1e-6)
        print(f"\n🎯 Gerçek Çözümle Karşılaştırma:")
        print(f"   True Solution: {true_solution}")
        print(f"   Absolute Error (MAE): {error:.4f}")
        print(f"   Relative Error: {relative_error:.4f} ({relative_error*100:.2f}%)")
    else:
        print("\n⚠️  Gerçek çözüm dosyası bulunamadı")
    
    print(f"\n🎉 Inference tamamlandı!")

if __name__ == "__main__":
    main()