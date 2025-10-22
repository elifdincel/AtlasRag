# 🧭 AtlasRAG • Türkçe Bilgi Asistanı (Akbank GAIH Final Projesi)

Bu repo, **RAG tabanlı bir Türkçe soru‑cevap chatbot** uygulamasıdır.  
Amaç: Belgelere dayalı, **kaynaklı ve güvenli** yanıtlar üretmek.

## 🎯 Proje Özeti
- **RAG mimarisi:** Sentence‑Transformers embedding → FAISS vektör arama → MMR re‑ranking → Flan‑T5 jenerasyon (opsiyonel Gemini).
- **Veri seti:** [Metin/WikiRAG‑TR](https://huggingface.co/datasets/Metin/WikiRAG-TR) (6k satır). İsterseniz kendi `.txt`/`.pdf` içeriklerinizi de ekleyebilirsiniz.
- **Arayüz:** Streamlit (chat + ayarlar + mini‑eval).
- **Teslim kriterlerini karşılar:** README, veri seti bilgisi, çalışma kılavuzu, mimari açıklama, web arayüzü. 

## 🗂️ Proje Yapısı
```
app.py              # Streamlit arayüzü + RAG boru hattı
eval.py             # Mini değerlendirme (hit@k)
requirements.txt    # Bağımlılıklar
.env.example        # Opsiyonel Gemini anahtarı
eval.tsv            # 20 örnek soru (örnek)
index/              # FAISS indeks dosyaları (ilk çalıştırma sonrası)
data/               # İsterseniz kendi belgeleriniz
```

## 🧰 Kurulum (lokal / Colab / Kaggle)
```bash
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
# (Opsiyonel) Gemini için .env dosyasına GOOGLE_API_KEY=... ekleyin
streamlit run app.py
```
Tarayıcı: http://localhost:8501

## 🌐 Deploy
- **Hugging Face Spaces (Streamlit):**
  - New Space → SDK: “Streamlit” → `app.py` ve `requirements.txt` yükleyin
  - (Opsiyonel) “Secrets” kısmına `GOOGLE_API_KEY` ekleyin
- **Railway/Render/Heroku:** `streamlit run app.py` komutunu çalıştıracak bir süreç tanımlayın.

## 🧪 Mini‑Eval
- `eval.py` dosyası 20 soru üzerinden basit bir **hit@k** metriği hesaplar:
```bash
python eval.py --eval_path eval.tsv --k 5
```

## 🧱 Mimarinin Ayrıntıları
- **Embedding:** `paraphrase-multilingual-MiniLM-L12-v2` (hafif, Türkçe için iyi).
- **Vektör DB:** `faiss.IndexFlatIP` (normalize edilmiş vektörlerle inner‑product = cosine).
- **MMR re‑ranking:** benzerlik + çeşitlilik dengesi (λ varsayılan 0.5).
- **Jenerasyon:** `google/flan-t5-base` (CPU'da çalışır), opsiyonel **Gemini 1.5**.
- **Prompt güvenliği:** “Sadece bağlamdan yararlan, emin değilsen söyle.” kuralı.

## 📊 Ekran Görüntüleri (öneri)
- Ana sohbet ekranı
- Kaynak listesi bloğu
- Sidebar ayarları
- Mini‑eval sonucu (json)

## ✅ PDF Kriterleri ile Eşleşme
- **GitHub & README:** Bu belge + açıklamalar ✔️  
- **Veri Seti:** WikiRAG‑TR açıklaması, nasıl kullanıldığı ✔️  
- **Çalışma Kılavuzu:** Kurulum/çalıştırma adımları ✔️  
- **Mimari:** RAG bileşenleri + şema açıklaması ✔️  
- **Web Arayüzü:** Streamlit chat + ayarlar + mini‑eval ✔️

## ⚠️ Notlar
- İlk indeksleme (4k satır) birkaç dakika sürebilir; sonrasında cache edilir.
- Tüm veri ile çalıştırmak isterseniz sidebar’dan “Tümü” seçeneğini açın.
- Büyük modeller yerine hafif seçenekler kullanıldı (lisans seviyesinde hızlı demo).

## 👩‍⚖️ Lisans
MIT