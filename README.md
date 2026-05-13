# Hatay Kadın Kooperatifi - Yapay Zeka Müşteri İletişim Otomasyonu

Bu proje, küçük ölçekli kooperatiflerin sipariş, stok ve müşteri iletişim süreçlerini otomatize etmek için geliştirilmiş Ajan Tabanlı (Agent-based) bir chatbot çözümüdür.

## Kullanılan Yapay Zeka Yaklaşımı
Projemizde statik ve sadece metin üreten sistemler yerine, aksiyon alabilen otonom bir mimari kullanılmıştır:
* Doğal Dil İşleme: Müşteri taleplerini anlama ve bağlama uygun, yöresel bir dille yanıt üretme.
* RAG (Retrieval-Augmented Generation): Ürün içerikleri ve iade koşulları gibi sabit bilgiler için dışarıdan veri okuma sistemi.
* AI Agent (Ajan Mimarisi): Sistem, müşterinin kargo veya stok durumu sorması halinde anlık veritabanına bağlanarak güncel durumu çeken fonksiyonlara (Tools) sahiptir.

## Sistem Mimarisi
1. Kullanıcı Arayüzü: Müşterinin iletişim kurduğu web platformu (Geliştirme Aşamasında).
2. Yapay Zeka Katmanı: LangChain framework'ü ile oluşturulmuş, yönlendirme ve aksiyon alma mantığını yürüten Agent.
3. Veri Katmanı: Sabit bilgiler için TXT dosyası, dinamik sipariş/stok verileri için simüle edilmiş JSON veritabanı.
