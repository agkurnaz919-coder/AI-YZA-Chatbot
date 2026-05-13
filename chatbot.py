import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# AYARLAR VE API
# ==========================================
# Kendi OpenAI API anahtarını buraya eklemelisin
os.environ["OPENAI_API_KEY"] = "sk-senin-api-anahtarin-buraya"

# ==========================================
# AŞAMA 1: RAG SİSTEMİ (TXT Dosyasını Okuma)
# ==========================================
try:
    loader = TextLoader("urun_bilgileri.txt", encoding="utf-8")
    dokumanlar = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    bolunmus_dokumanlar = text_splitter.split_documents(dokumanlar)

    vectorstore = FAISS.from_documents(bolunmus_dokumanlar, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    rag_araci = create_retriever_tool(
        retriever,
        "urun_ve_kosul_bilgisi_getir",
        "Hatay kooperatif ürünlerinin içerikleri, yapılışları, kullanım alanları, saklama koşulları veya iade/kargo politikaları hakkında bilgi ararken KESİNLİKLE bu aracı kullan."
    )
    print("✅ RAG Sistemi (Bilgi Bankası) başarıyla yüklendi.")
except Exception as e:
    print(f"❌ RAG Yükleme Hatası: urun_bilgileri.txt dosyası bulunamadı veya okunamadı. ({e})")

# ==========================================
# AŞAMA 2: DİNAMİK ARAÇLAR (JSON Dosyasını Okuma)
# ==========================================

@tool
def kargo_durumu_sorgula(siparis_no: str) -> str:
    """Kullanıcı siparişinin durumunu, nerede olduğunu veya ne zaman teslim edileceğini sorduğunda bu aracı çalıştır."""
    try:
        with open("veritabani.json", "r", encoding="utf-8") as f:
            veritabani = json.load(f)
            
        siparis = veritabani["siparisler"].get(siparis_no)
        if siparis:
            return f"Sipariş {siparis_no} bilgileri: Ürün: {siparis['urun']}, Durum: {siparis['durum']}, Tahmini Teslimat: {siparis['teslimat']}, Kargo Firması: {siparis['kargo_firmasi']}."
        else:
            return f"{siparis_no} numaralı sipariş sistemimizde bulunamadı. Lütfen numarayı kontrol etmesini isteyin."
    except Exception as e:
        return "Veritabanına ulaşılamıyor, lütfen daha sonra tekrar deneyin."

@tool
def stok_ve_fiyat_sorgula(urun_kodu: str) -> str:
    """Kullanıcı bir ürünün stokta olup olmadığını veya fiyatını sorduğunda bu aracı çalıştır. 
    urun_kodu olarak sadece 'nar_eksisi' veya 'surk_peyniri' kullanabilirsin."""
    try:
        with open("veritabani.json", "r", encoding="utf-8") as f:
            veritabani = json.load(f)
            
        urun = veritabani["stoklar"].get(urun_kodu)
        if urun:
            return f"Ürün: {urun['isim']}, Güncel Stok: {urun['adet']} adet, Fiyat: {urun['fiyat_tl']} TL."
        else:
            return "Bu ürün şu an stoklarımızda bulunmamaktadır."
    except Exception as e:
        return "Veritabanına ulaşılamıyor."

tools = [rag_araci, kargo_durumu_sorgula, stok_ve_fiyat_sorgula]

# ==========================================
# AŞAMA 3: AJAN (AGENT) KURULUMU
# ==========================================

# Dil modelini tanımlıyoruz
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Chatbotun kişiliğini ve çalışma kurallarını belirliyoruz
prompt = ChatPromptTemplate.from_messages([
    ("system", """Sen Hatay Kadın Kooperatifi'nin resmi dijital asistanısın. 
    Amacın müşterilere yöresel ürünler hakkında bilgi vermek, stok durumunu bildirmek ve sipariş/kargo süreçlerini takip etmektir.
    Çok nazik, yardımsever ve yöresel bir sıcaklıkta konuşmalısın. Gerekirse konuşmalarına samimi emojiler ekleyebilirsin.
    
    KURALLAR:
    1. Ürün bilgileri, iade koşulları ve saklama talimatları için DAİMA 'urun_ve_kosul_bilgisi_getir' aracını kullan.
    2. Kargo veya sipariş durumu sorulduğunda müşteriden sipariş numarasını iste ve 'kargo_durumu_sorgula' aracını kullan.
    3. Müşteri fiyat veya elinizde ürün olup olmadığını (stok) sorarsa 'stok_ve_fiyat_sorgula' aracını kullan. Hangi ürünü kastettiğini anla (nar_eksisi veya surk_peyniri).
    4. Kendi kendine bilgi uydurma. Veritabanında veya bilgi bankasında olmayan bir şey sorulursa kibarca kooperatif yetkilisine yönlendir."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True jüriye arka planda dönen işlemleri göstermek için harikadır!

# ==========================================
# AŞAMA 4: CANLI TEST EKRANI
# ==========================================

def chat_baslat():
    print("\n" + "="*50)
    print("Hatay Kadın Kooperatifi Chatbot'u Başlatıldı!")
    print("Çıkmak için 'q' veya 'çıkış' yazabilirsiniz.")
    print("="*50 + "\n")
    
    while True:
        kullanici_mesaji = input("Müşteri: ")
        if kullanici_mesaji.lower() in ['q', 'çıkış', 'cikis', 'quit']:
            print("Chatbot kapatılıyor. İyi günler!")
            break
            
        try:
            # Ajanı çalıştırıp cevabı alıyoruz
            response = agent_executor.invoke({"input": kullanici_mesaji})
            print(f"\nChatbot: {response['output']}\n" + "-"*50 + "\n")
        except Exception as e:
            print(f"\nBir hata oluştu: {e}\n")

if __name__ == "__main__":
    # Konsol üzerinden canlı sohbeti başlat
    chat_baslat()