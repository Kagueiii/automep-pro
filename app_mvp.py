import streamlit as st
import fitz
import tempfile
import os
import hashlib
import pandas as pd
import google.generativeai as genai
import io
import re
from fpdf import FPDF

# Importações do nosso motor core (Garantindo quebra de linha)
from core_engine import IngestorPDFLocal
from core_engine import TriagemEspacialLocal

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="AutoMEP Pro | Engine Espacial", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

# 2. INJEÇÃO DE CSS
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 95%; }
        div[data-testid="metric-container"] {
            background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px;
            border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border-left: 5px solid #1E3A8A;
        }
        .main-title { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 0px; }
        .subtitle { font-size: 1.1rem; color: #64748b; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

if "laudo_salvo" not in st.session_state: st.session_state["laudo_salvo"] = None
if "pdf_hash_atual" not in st.session_state: st.session_state["pdf_hash_atual"] = None

@st.cache_data(show_spinner=False)
def processar_auditoria(file_id: str, _file_bytes: bytes, sensibilidade: float, disciplina: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        caminho_tmp = tmp.name
    try:
        disc_mestra = disciplina if disciplina != "Interdisciplinar" else None
        ingestor = IngestorPDFLocal()
        componentes, w, h = ingestor.processar_pdf(caminho_tmp, disciplina_mestra=disc_mestra)
        motor = TriagemEspacialLocal(limite_conflito_px=sensibilidade)
        clusters_ricos = motor.executar_triagem(componentes, w, h, disciplina_mestra=disc_mestra)
        
        doc = fitz.open(caminho_tmp)
        pagina = doc.load_page(0)
        for cluster in clusters_ricos:
            for id_node in cluster["ids"]:
                comp = motor.mapa_componentes[id_node]
                b = comp.bbox_relativo
                rect = fitz.Rect(b[0]*w, b[1]*h, b[2]*w, b[3]*h)
                cor = (1, 0, 0) if cluster["status"] == "Crítico" else (1, 0.5, 0)
                pagina.draw_rect(rect, color=cor, width=2, fill_opacity=0.2)
        
        pix = pagina.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()

        dados_tabela = []
        for i, c in enumerate(clusters_ricos):
            dados_tabela.append({
                "ID do Clash": f"CLASH-{i+1:03d}",
                "Categoria": c["categoria_conflito"],
                "Status Técnica": c["status"],
                "Descrição Técnica": c["descricao_tecnica"]
            })
        return img_bytes, dados_tabela, len(componentes), w, h
    finally:
        if os.path.exists(caminho_tmp): os.remove(caminho_tmp)

st.markdown('<p class="main-title">🏗️ AutoMEP Pro <sub>v1.0</sub></p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">LTS Production Build | Engine Espacial de Precisão</p>', unsafe_allow_html=True)

with st.sidebar:
    arquivo_pdf = st.file_uploader("Upload de Prancha (.PDF)", type=["pdf"])
    st.markdown("---")
    sensibilidade = st.slider("Sensibilidade (px)", 0.5, 5.0, 1.0, 0.5)
    disciplina = st.selectbox("Modo de Auditoria", ["Interdisciplinar", "eletrica", "hidraulica"])

if arquivo_pdf:
    file_bytes = arquivo_pdf.read()
    file_id = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state["pdf_hash_atual"] != file_id:
        st.session_state["laudo_salvo"] = None
        st.session_state["pdf_hash_atual"] = file_id
    
    with st.spinner("⏳ Triangulando metadados espaciais..."):
        img_bytes, dados, num_comp, w, h = processar_auditoria(file_id, file_bytes, sensibilidade, disciplina)
        df = pd.DataFrame(dados)

        m1, m2, m3 = st.columns(3)
        m1.metric("Componentes", num_comp)
        m2.metric("Hard Clashes", len(df[df['Categoria'].str.contains('HARD')]))
        m3.metric("Overlaps", len(df[df['Categoria'].str.contains('OVERLAP')]))

        tab1, tab2, tab3 = st.tabs(["🗺️ Mapa Espacial", "📑 Log Técnico", "🤖 Laudo da IA"])
        with tab1: st.image(img_bytes, use_container_width=True)
        with tab2: st.dataframe(df, use_container_width=True, hide_index=True)
        with tab3:
            def gerar_pdf_bytes(texto):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Relatorio de Compatibilizacao AutoMEP Pro", ln=True, align="C")
                pdf.set_font("Arial", size=12)
                t_limpo = re.sub(r'[*#`]+', '', texto).replace('\t', '    ')
                m_tipografico = {'•': '-', '“': '"', '”': '"', '—': '-', '–': '-'}
                for k, v in m_tipografico.items(): t_limpo = t_limpo.replace(k, v)
                f_final = []
                for linha in t_limpo.split('\n'):
                    l = linha.lstrip()
                    f_final.append('\n' + l if 'CLASH-' in l or l.endswith(':') else l)
                pdf.multi_cell(0, 6, '\n'.join(f_final).encode('latin-1', 'ignore').decode('latin-1'))
                return pdf.output()

            if st.session_state["laudo_salvo"]:
                st.markdown(st.session_state["laudo_salvo"])
                st.download_button("📥 Baixar PDF Oficial", data=gerar_pdf_bytes(st.session_state["laudo_salvo"]), file_name="Relatorio_AutoMEP.pdf")
                if st.button("🧹 Limpar Cache"): 
                    st.session_state["laudo_salvo"] = None
                    st.rerun()
            elif st.button("✨ Gerar Laudo Unificado"):
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                batch_str = ""
                for _, r in df.head(15).iterrows():
                    batch_str += f"- **{r['ID do Clash']}** ({r['Categoria']}) | Status: {r['Status Técnica']}\n  - Contexto: {r['Descrição Técnica']}\n"
                
                prompt = f"""Você é um Engenheiro Sênior MEP. Analise as interferências detectadas:
{batch_str}
PROIBIDO usar tabelas Markdown. Use apenas listas com hifens, negritos simples e subtítulos claros. 
Forneça Risco, Análise Técnica e Solução para cada item."""
                
                res = model.generate_content(prompt, stream=True)
                def iter_seguro(s):
                    for chunk in s:
                        try: yield chunk.text
                        except ValueError: yield "\n*[Filtro de Segurança]*\n"
                
                st.session_state["laudo_salvo"] = st.write_stream(iter_seguro(res))
                st.rerun()
