import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pickle


class BipartiteGraphConvolution(K.Model):
    """
    Kısmi iki parçalı graf evrişimi (soldan sağa veya sağdan sola).
    Bu katman, kısıt düğümleri ile değişken düğümleri arasındaki mesaj geçişini yönetir.
    """
    def __init__(self, emb_size, activation, initializer, right_to_left=False, name=None):
        super().__init__(name=name)
        
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left

        # Özellik katmanları: Gelen düğüm özelliklerini gömme boyutuna dönüştürür.
        # Sol düğümler (constraints - kısıtlar) veya Sağ düğümler (variables - değişkenler)
        self.feature_module_left = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=True, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name=f"{name}_feat_left")

        # Kenar özellikleri katmanı: Kenar özelliklerini gömme boyutuna dönüştürür.
        # Bu, A_ineq matrisindeki A_ij katsayılarının özellikleridir.
        self.feature_module_edge = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name=f"{name}_feat_edge")

        # Sağ düğümler (variables - değişkenler) veya Sol düğümler (constraints - kısıtlar)
        self.feature_module_right = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name=f"{name}_feat_right")

        # Çıkış katmanı: Toplanan mesajları önceki düğüm özellikleriyle birleştirir
        # ve nihai düğüm özelliklerini üretir.
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name=f"{name}_output")
    
    def build(self, input_shapes):
        # input_shapes: (receiving_node_features_shape, edge_indices_shape, edge_features_shape, sending_node_features_shape, scatter_out_size_val)
        receiving_shape, ei_shape, ev_shape, sending_shape, _ = input_shapes 

        self.feature_module_left.build(receiving_shape) # Eğer left_features receiving ise
        self.feature_module_edge.build((None, ev_shape[1])) # Kenar özelliklerinin ham boyutu
        self.feature_module_right.build(sending_shape) # Eğer right_features sending ise

        # output_module'un girdisi: tf.concat([conv_output, prev_features], axis=1)
        # conv_output boyutu: (scatter_out_size, self.emb_size)
        # prev_features boyutu: (scatter_out_size, self.emb_size) (çünkü bunlar da embedding alanına dönüştürülür)
        # Yani birleştirilmiş boyut: (None, 2 * self.emb_size)
        self.output_module.build([None, 2 * self.emb_size]) 
        self.built = True
    
    def call(self, inputs):
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left: # Sağdan sola (değişkenden kısıta) mesaj geçişi
            scatter_dim = 0 # Sol düğümlere (kısıtlar) dağıt
            processed_sending_features = self.feature_module_right(right_features) # Gönderen: değişkenler
            prev_features_for_concat = left_features # Alıcı: kısıtlar (raw input)
        else: # Soldan sağa (kısıttan değişkene) mesaj geçişi
            scatter_dim = 1 # Sağ düğümlere (değişkenler) dağıt
            processed_sending_features = self.feature_module_left(left_features) # Gönderen: kısıtlar
            prev_features_for_concat = right_features # Alıcı: değişkenler (raw input)

        # Kenar özelliklerini işle
        processed_edge_features = self.feature_module_edge(edge_features)

        # Ortak özellikleri hesapla: kenar_özellikleri * gönderen_düğüm_özellikleri
        # tf.gather, edge_indices'e göre gönderen düğüm özelliklerini seçer
        if scatter_dim == 0: # Değişkenden kısıta
            joint_features = processed_edge_features * tf.gather(
                processed_sending_features,
                axis=0,
                indices=edge_indices[1] # edge_indices[1] sağ düğüm (değişken) indekslerini içerir
            )
        else: # Kısıttan değişkene
            joint_features = processed_edge_features * tf.gather(
                processed_sending_features,
                axis=0,
                indices=edge_indices[0] # edge_indices[0] sol düğüm (kısıt) indekslerini içerir
            )

        # Evrişim (toplama) işlemi
        # Mesajları alıcı düğümlere toplar
        conv_output = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
            shape=[scatter_out_size, self.emb_size]
        )

        # Önceki özelliklerle birleştir (kalıntı bağlantı veya birleştirme)
        # output_module, tf.concat([conv_output, prev_features_for_concat]) bekler
        output = self.output_module(tf.concat([conv_output, prev_features_for_concat], axis=1))

        return output


class VariableToVariableConvolution(K.Model):
    """
    Karesel terim etkileşimleri (H matrisi) için değişkenler arası evrişim.
    """
    def __init__(self, emb_size, activation, initializer, name=None):
        super().__init__(name=name)
        
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        
        # Değişken düğümleri için özellik işleme
        self.feature_module_var = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name=f"{name}_feat_var")
        
        # Q-kenar özellikleri için özellik işleme (H matrisindeki katsayılar)
        self.feature_module_edge = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name=f"{name}_feat_qedge")
        
        # Çıkış modülü
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name=f"{name}_output")
    
    def build(self, input_shapes):
        # input_shapes: (var_features_shape, edge_indices_shape, edge_features_shape, n_vars_total_val)
        var_shape, edge_indices_shape, edge_features_shape, _ = input_shapes
        
        self.feature_module_var.build(var_shape)
        self.feature_module_edge.build((None, edge_features_shape[1])) # Kenar özelliklerinin ham boyutu
        
        # output_module'un girdisi: tf.concat([conv_output, var_features], axis=1)
        # conv_output boyutu: (n_vars_total, self.emb_size)
        # var_features boyutu: (n_vars_total, var_shape[1]) (ham özellikler)
        self.output_module.build([None, self.emb_size + var_shape[1]]) 
        self.built = True
    
    def call(self, inputs):
        var_features, edge_indices, edge_features, n_vars_total = inputs
        
        # Düğüm özelliklerini işle
        processed_var_features = self.feature_module_var(var_features)
        
        # Q-kenar özelliklerini işle
        processed_edge_features = self.feature_module_edge(edge_features)

        # Ortak özellikleri hesapla: kenar_özellikleri * gönderen_düğüm_özellikleri
        # Burada gönderen düğümler edge_indices[1]'deki düğümlerdir (Q matrisi simetrikse fark etmez)
        joint_features = processed_edge_features * tf.gather(
            processed_var_features,
            axis=0,
            indices=edge_indices[1] 
        )
        
        # Evrişim (toplama) işlemi alıcı düğümlere (edge_indices[0])
        conv_output = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[0], axis=1),
            shape=[n_vars_total, self.emb_size]
        )
        
        # Önceki (ham) özelliklerle birleştir (kalıntı bağlantı)
        output = self.output_module(tf.concat([conv_output, var_features], axis=1))
        
        return output


class QPGNNPolicy(K.Model):
    """
    Karesel Programlama (QP) için Graf Evrişimsel Sinir Ağı modeli.
    Düğüm seviyesi veya graf seviyesi çıktıları destekler.
    Her aktivasyondan sonra isteğe bağlı dropout içerir.

    Bu model, aşağıdaki görevler için kullanılabilir:
    1. Fizibilite Tahmini (is_graph_level=True, output_units=1, output_activation='sigmoid')
    2. Optimal Amaç Değeri Tahmini (is_graph_level=True, output_units=1, output_activation=None)
    3. Optimal Çözüm Tahmini (is_graph_level=False, output_units=n_vars, output_activation=None)
    """
    def __init__(
        self,
        emb_size,
        cons_nfeats,  # Kısıt düğümleri için özellik sayısı (örn., 2: [b, tip])
        edge_nfeats,  # Kısıt-değişken kenarları için özellik sayısı (örn., 1: A_ij)
        var_nfeats,   # Değişken düğümleri için özellik sayısı (örn., 3: [c, lb, ub])
        qedge_nfeats=1, # Değişken-değişken kenarları için özellik sayısı (örn., 1: H_ij)
        is_graph_level=True, # True ise graf seviyesi (tek değer); False ise düğüm seviyesi (vektör)
        output_units=1, # Son katman için çıktı birimi sayısı (skaler için 1, çözüm vektörü için n_vars)
        output_activation=None, # Son çıktı katmanı için aktivasyon (örn., fizibilite için 'sigmoid')
        dropout_rate=0.0,
        name="QPGNNPolicy"
    ):
        super().__init__(name=name)

        self.emb_size = emb_size
        self.cons_nfeats = cons_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = var_nfeats
        self.qedge_nfeats = qedge_nfeats
        self.is_graph_level = is_graph_level
        self.output_units = output_units
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.activation = K.activations.relu # Ara katmanlar için Relu
        self.initializer = K.initializers.Orthogonal()

        # Dropout katmanı
        self.dropout_layer = K.layers.Dropout(self.dropout_rate)

        # GÖMME KATMANLARI: Ham özellikleri gömme alanına dönüştürür.
        self.cons_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ], name="cons_embedding")

        self.edge_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name="edge_embedding")
        
        self.qedge_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
        ], name="qedge_embedding")

        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
        ], name="var_embedding")

        # GRAF EVRİŞİM KATMANLARI - Kısıt-Değişken Etkileşimleri (ilk katman)
        # Değişkenden Kısıta (sağdan sola) mesaj
        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer,
                                                     right_to_left=True, name="conv_v_to_c_L1")
        # Kısıttan Değişkene (soldan sağa) mesaj
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer,
                                                     right_to_left=False, name="conv_c_to_v_L1")

        # GRAF EVRİŞİM KATMANLARI - İkinci katman
        self.conv_v_to_c2 = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer,
                                                      right_to_left=True, name="conv_v_to_c_L2")
        self.conv_c_to_v2 = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer,
                                                      right_to_left=False, name="conv_c_to_v_L2")
        
        # DEĞİŞKEN-DEĞİŞKEN EVRİŞİM KATMANLARI - Karesel terimler için (ilk katman)
        self.conv_v_to_v = VariableToVariableConvolution(self.emb_size, self.activation, self.initializer,
                                                        name="conv_v_to_v_L1")
        self.conv_v_to_v2 = VariableToVariableConvolution(self.emb_size, self.activation, self.initializer,
                                                         name="conv_v_to_v_L2")

        # ÇIKIŞ MODÜLÜ: Nihai düğüm gömmelerini istenen çıktıya dönüştürür.
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.output_units, activation=self.output_activation, kernel_initializer=self.initializer),
        ], name="final_output_module")
        
        # Modelin ağırlıklarını oluşturmak için build metodunu hemen çağır.
        # input_shapes: [cons_feats, edge_indices, edge_feats, var_feats, qedge_indices, qedge_feats, n_cons_total, n_vars_total, n_cons_small, n_vars_small]
        self.build([
            (None, self.cons_nfeats),       # Kısıt özellikleri (örn., (None, 2))
            (2, None),                      # Kısıt-değişken kenar indeksleri (örn., (2, num_A_edges))
            (None, self.edge_nfeats),       # Kısıt-değişken kenar özellikleri (örn., (None, 1))
            (None, self.var_nfeats),        # Değişken özellikleri (örn., (None, 3))
            (2, None),                      # Değişken-değişken kenar indeksleri (örn., (2, num_Q_edges))
            (None, self.qedge_nfeats),      # Değişken-değişken kenar özellikleri (örn., (None, 1))
            (None, ),                       # n_cons_total (batch_size * num_cons_per_graph)
            (None, ),                       # n_vars_total (batch_size * num_vars_per_graph)
            (None, ),                       # n_cons_small (tek bir grafikteki kısıt düğümü sayısı)
            (None, ),                       # n_vars_small (tek bir grafikteki değişken düğümü sayısı)
        ])
        
        # Kaydetme/geri yükleme için değişken adlarını topolojik sırada sakla
        self.variables_topological_order = [v.name for v in self.variables]
    
    def build(self, input_shapes):
        # input_shapes'i, call metoduna nasıl iletildiğine göre açar
        c_feat_shape, ei_shape, ev_feat_shape, v_feat_shape, qi_shape, qv_feat_shape, _, _, _, _ = input_shapes
        
        # Gömme katmanlarını oluştur
        self.cons_embedding.build(c_feat_shape)
        self.edge_embedding.build((None, ev_feat_shape[1]))
        self.qedge_embedding.build((None, qv_feat_shape[1]))
        self.var_embedding.build(v_feat_shape)
        
        # Graf evrişim katmanlarını oluştur
        # BipartiteGraphConvolution.build'e giriş: (receiving_feat_shape, edge_indices_shape, embedded_edge_feat_shape, sending_feat_shape, scatter_out_size)
        # embedded_edge_feat_shape her zaman (None, self.emb_size) olmalıdır
        emb_shape = (None, self.emb_size)

        self.conv_v_to_c.build((c_feat_shape, ei_shape, emb_shape, emb_shape, None)) # right_to_left=True, receiving=cons, sending=var
        self.conv_c_to_v.build((emb_shape, ei_shape, emb_shape, v_feat_shape, None)) # right_to_left=False, receiving=var, sending=cons

        self.conv_v_to_c2.build((c_feat_shape, ei_shape, emb_shape, emb_shape, None))
        self.conv_c_to_v2.build((emb_shape, ei_shape, emb_shape, v_feat_shape, None))
        
        # VariableToVariableConvolution.build'e giriş: (var_feat_shape, edge_indices_shape, embedded_edge_feat_shape, n_vars_total_val)
        self.conv_v_to_v.build((emb_shape, qi_shape, emb_shape, None))
        self.conv_v_to_v2.build((emb_shape, qi_shape, emb_shape, None))
        
        # Çıkış modülünü oluştur
        if self.is_graph_level:
            # Graf seviyesi için, havuzlanmış değişken ve kısıt özelliklerini birleştiririz.
            # Her biri havuzlamadan sonra (None, emb_size) olduğundan, birleşmiş boyut (None, 2 * emb_size) olur.
            self.output_module.build([None, 2 * self.emb_size])
        else:
            # Düğüm seviyesi için, doğrudan değişken özelliklerini (None, emb_size) kullanırız.
            self.output_module.build(emb_shape) 

        self.built = True # built bayrağını ayarla
        
    def call(self, inputs, training=False):
        """
        Bir QP örneğini GNN üzerinden işler.
        training=True ise, dropout etkinleştirilir.
        """
        constraint_features_raw, edge_indices_A, edge_features_A_raw, variable_features_raw, \
        qedge_indices_H, qedge_features_H_raw, n_cons_total, n_vars_total, n_cons_small, n_vars_small = inputs

        # GÖMME KATMANLARI (Ham özellikleri gömme alanına dönüştürür)
        # Bunlar başlangıçta bir kez uygulanır.
        constraint_features = self.cons_embedding(constraint_features_raw)
        edge_features = self.edge_embedding(edge_features_A_raw)
        qedge_features = self.qedge_embedding(qedge_features_H_raw)
        variable_features = self.var_embedding(variable_features_raw)

        #----------------------------------------------------------------
        # İLK EVRİŞİM KATMANI
        #----------------------------------------------------------------
        # Kısıt-Değişken iki parçalı evrişim (Değişkenden Kısıta Mesaj)
        constraint_features = self.conv_v_to_c((
            constraint_features, edge_indices_A, edge_features, variable_features, n_cons_total))
        constraint_features = self.activation(constraint_features)
        if training and self.dropout_rate > 0:
            constraint_features = self.dropout_layer(constraint_features, training=training)

        # Değişken-Kısıt iki parçalı evrişim (Kısıttan Değişkene Mesaj)
        variable_features = self.conv_c_to_v((
            constraint_features, edge_indices_A, edge_features, variable_features, n_vars_total))
        variable_features = self.activation(variable_features)
        if training and self.dropout_rate > 0:
            variable_features = self.dropout_layer(variable_features, training=training)
        
        # Değişken-Değişken evrişim (karesel terimler için)
        variable_features = self.conv_v_to_v((
            variable_features, qedge_indices_H, qedge_features, n_vars_total))
        variable_features = self.activation(variable_features)
        if training and self.dropout_rate > 0:
            variable_features = self.dropout_layer(variable_features, training=training)

        #----------------------------------------------------------------
        # İKİNCİ EVRİŞİM KATMANI
        #----------------------------------------------------------------
        # Kısıt-Değişken iki parçalı evrişim
        constraint_features = self.conv_v_to_c2((
            constraint_features, edge_indices_A, edge_features, variable_features, n_cons_total))
        constraint_features = self.activation(constraint_features)
        if training and self.dropout_rate > 0:
            constraint_features = self.dropout_layer(constraint_features, training=training)

        # Değişken-Kısıt iki parçalı evrişim
        variable_features = self.conv_c_to_v2((
            constraint_features, edge_indices_A, edge_features, variable_features, n_vars_total))
        variable_features = self.activation(variable_features)
        if training and self.dropout_rate > 0:
            variable_features = self.dropout_layer(variable_features, training=training)
        
        # Değişken-Değişken evrişim (karesel terimler için)
        variable_features = self.conv_v_to_v2((
            variable_features, qedge_indices_H, qedge_features, n_vars_total))
        variable_features = self.activation(variable_features)
        if training and self.dropout_rate > 0:
            variable_features = self.dropout_layer(variable_features, training=training)
        
        #----------------------------------------------------------------
        # ÇIKIŞ KATMANI
        #----------------------------------------------------------------
        if self.is_graph_level:
            # Graf seviyesi çıktı: Değişken ve kısıt özelliklerini havuzla
            # Giriş özellikleri zaten toplu veri kümeleri için birleştirilmiştir: (batch_size * num_nodes_per_graph, emb_size)
            # Ortalama havuzlama için (batch_size, num_nodes_per_graph, emb_size) şeklinde yeniden şekillendirmek gerekir.
            
            # Değişken özelliklerini yeniden şekillendir: (batch_size, n_vars_small, emb_size)
            batch_size_vars = tf.cast(n_vars_total / n_vars_small, tf.int32)
            variable_features_reshaped = tf.reshape(variable_features, [batch_size_vars, n_vars_small, self.emb_size])
            variable_features_mean = tf.reduce_mean(variable_features_reshaped, axis=1) # Her grafikteki değişkenler üzerinde ortalama havuzlama

            # Kısıt özelliklerini yeniden şekillendir: (batch_size, n_cons_small, emb_size)
            batch_size_cons = tf.cast(n_cons_total / n_cons_small, tf.int32)
            constraint_features_reshaped = tf.reshape(constraint_features, [batch_size_cons, n_cons_small, self.emb_size])
            constraint_features_mean = tf.reduce_mean(constraint_features_reshaped, axis=1) # Her grafikteki kısıtlar üzerinde ortalama havuzlama

            final_features = tf.concat([variable_features_mean, constraint_features_mean], axis=1) # Havuzlanmış özellikleri birleştir
            
            # Dropout (isteğe bağlı)
            if training and self.dropout_rate > 0:
                final_features = self.dropout_layer(final_features, training=training)
        else:
            # Düğüm seviyesi çıktı: Doğrudan değişken özelliklerini kullan
            # Bu genellikle değişken başına çözüm tahminleri için kullanılır.
            final_features = variable_features
            if training and self.dropout_rate > 0:
                final_features = self.dropout_layer(final_features, training=training)

        output = self.output_module(final_features)
        return output
        
    def save_state(self, path):
        """Modelin ağırlıklarını kaydeder."""
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    def restore_state(self, path):
        """Modelin ağırlıklarını geri yükler."""
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))