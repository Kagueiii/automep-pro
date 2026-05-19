import streamlit as st
import fitz
import tempfile
import os
import hashlib
import pandas as pd
import google.generativeai as genai
import io
import re
import json
from fpdf import FPDF

from core_engine import IngestorPDFLocal
from core_engine import TriagemEspacialLocal

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E CSS
# ==========================================
st.set_page_config(page_title="AutoMEP Pro | Cascata de IA", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

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

# ==========================================
# FASE 1: MOTOR GEOMÉTRICO LOCAL
# ==========================================
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

# ==========================================
# FASE 2: AGENTE DE TRIAGEM (Filtro Semântico)
# ==========================================
@st.cache_data(show_spinner=False)
def filtrar_falsos_positivos_ia(dados_tabela: list, api_key: str) -> list:
    if not dados_tabela: return []
    todos_ids = [d["ID do Clash"] for d in dados_tabela]
    
    try:
        genai.configure(api_key=api_key)
        # IA Barata para filtrar lixo (Flash)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        payload_contexto = [{"ID": d["ID do Clash"], "Desc": d["Descrição Técnica"], "Cat": d["Categoria"]} for d in dados_tabela]
        
        prompt_triagem = f"""Você atua como um Filtro Semântico de Engenharia. 
Identifique e remova falsos positivos (ex: ruído de OCR, textos de margem/carimbo, ou elementos que não são colisões físicas reais).
Mantenha apenas os erros físicos e colisões graves.

Lista de Conflitos:
{json.dumps(payload_contexto, ensure_ascii=False)}

REGRAS: 
Devolva ESTRITAMENTE um array JSON com os 'ID' válidos."""

        resp = model.generate_content(
            prompt_triagem, 
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        
        ids_validados = json.loads(resp.text)
        if isinstance(ids_validados, list):
            return [str(i) for i in ids_validados]
        elif isinstance(ids_validados, dict):
            for val in ids_validados.values():
                if isinstance(val, list): return [str(i) for i in val]
        raise ValueError("Formato JSON inesperado.")
            
    except Exception as e:
        print(f"Fallback Stage 2 ativado: {e}")
        return todos_ids

# ==========================================
# 3. INTERFACE E ORQUESTRAÇÃO EM CASCATA
# ==========================================
st.markdown('<p class="main-title">🏗️ AutoMEP Pro <sub>v2.0</sub></p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Arquitetura em Cascata: Triagem Espacial e Auditoria de Compliance</p>', unsafe_allow_html=True)

with st.sidebar:
    arquivo_pdf = st.file_uploader("Upload de Prancha (.PDF)", type=["pdf"])
    st.markdown("---")
    sensibilidade = st.slider("Sensibilidade Espacial (px)", 0.5, 5.0, 1.0, 0.5)
    disciplina = st.selectbox("Modo de Auditoria", ["Interdisciplinar", "eletrica", "hidraulica"])

if arquivo_pdf:
    file_bytes = arquivo_pdf.read()
    file_id = hashlib.sha256(file_bytes).hexdigest()
    
    if st.session_state["pdf_hash_atual"] != file_id + str(sensibilidade) + disciplina:
        st.session_state["laudo_salvo"] = None
        st.session_state["pdf_hash_atual"] = file_id + str(sensibilidade) + disciplina
    
    try:
        api_key_segura = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("Erro Crítico: Chave 'GEMINI_API_KEY' não encontrada em `.streamlit/secrets.toml`.")
        st.stop()
    
    # --- FLUXO AUTÔNOMO: FASE 1 e 2 ---
    with st.spinner("⏳ Fase 1: Triangulando metadados espaciais (Motor Local)..."):
        img_bytes, dados_brutos, num_comp, w, h = processar_auditoria(file_id, file_bytes, sensibilidade, disciplina)
        df_bruto = pd.DataFrame(dados_brutos)

    with st.spinner('🤖 Fase 2 (IA de Triagem): Filtrando falsos positivos e ruído semântico...'):
        if not df_bruto.empty:
            ids_triados = filtrar_falsos_positivos_ia(dados_brutos, api_key_segura)
            df_limpo = df_bruto[df_bruto["ID do Clash"].isin(ids_triados)]
        else:
            df_limpo = pd.DataFrame(columns=["ID do Clash", "Categoria", "Status Técnica", "Descrição Técnica"])

    # Métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Componentes", num_comp)
    m2.metric("Conflitos Brutos", len(df_bruto))
    m3.metric("Erros Reais (Pós-Filtro)", len(df_limpo))
    m4.metric("Ruído Suprimido", f"{len(df_bruto) - len(df_limpo)}")

    tab1, tab2, tab3 = st.tabs(["🗺️ Mapa Espacial", "📑 Log de Triagem", "⚖️ Auditoria Sênior (IA)"])
    
    with tab1: st.image(img_bytes, use_container_width=True)
    with tab2: 
        if df_limpo.empty:
            st.success("✅ Nenhum erro real de engenharia detectado.")
        else:
            st.markdown("**Lista Purificada (Enviada para o Auditor Sênior)**")
            st.dataframe(df_limpo, use_container_width=True, hide_index=True)

    # --- FASE 3: AUDITOR SÊNIOR (Automático) ---
    with tab3:
        def gerar_pdf_bytes(texto):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Relatorio de Auditoria Compliance e Custos", ln=True, align="C")
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

        gerado_nesta_sessao = False

        # Condição de Automação Total: Se há erro, não há laudo ainda, GERE!
        if st.session_state["laudo_salvo"] is None and len(df_limpo) > 0:
            st.warning("⚠️ Erros detectados! Acionando Auditoria Sênior para precificação de risco...")
            try:
                genai.configure(api_key=api_key_segura)
                # Upgrade de IA: O Pro analisa leis e custos muito melhor
                model_auditor = genai.GenerativeModel("gemini-1.5-pro") 
                
                batch_str = ""
                # Limite de segurança: analisa no máximo os 15 erros mais graves para não estourar tokens
                for _, r in df_limpo.head(15).iterrows():
                    batch_str += f"- **{r['ID do Clash']}** ({r['Categoria']}) | Status Z: {r['Status Técnica']}\n  - Contexto: {r['Descrição Técnica']}\n"
                
                prompt_auditor = f"""Você é um Auditor Sênior MEP e Orçamentista Pessimista de Obras.
Analise esta lista de defeitos críticos purificados pelo nosso motor:

{batch_str}

Para CADA defeito, retorne:
1. Violação de Norma: Cite NBRs aplicáveis (ex: 5410, 8160, etc). Se contexto internacional, cite IBC ou NEC.
2. Falha Operacional: O que vai quebrar em 5 anos?
3. Custo de Retrabalho: Estime o pior cenário em R$ ou US$ (material + mão de obra ociosa).

REGRAS: PROIBIDO tabelas Markdown. Use listas e negrito simples. Seja frio e calculista."""
                
                resposta_stream = model_auditor.generate_content(prompt_auditor, stream=True)
                
                def iter_seguro(stream_resp):
                    for chunk in stream_resp:
                        try: yield chunk.text
                        except ValueError: yield "\n*[Filtro de Segurança]*\n"
                
                container_laudo = st.container()
                with container_laudo:
                    laudo_gerado = st.write_stream(iter_seguro(resposta_stream))
                    st.session_state["laudo_salvo"] = laudo_gerado
                    gerado_nesta_sessao = True # Sinaliza que o texto já está na tela

            except Exception as e_api:
                st.error(f"Falha na Auditoria Sênior: {e_api}")

        # Proteção de UX: Garante que o texto não desapareça se a página recarregar
        if st.session_state["laudo_salvo"]:
            # Se o texto já estava salvo e NÃO foi gerado agora, imprime com markdown estático
            if not gerado_nesta_sessao:
                st.markdown(st.session_state["laudo_salvo"])
                
            pdf_bytes = gerar_pdf_bytes(st.session_state["laudo_salvo"])
            st.download_button("📥 Baixar Parecer Técnico (PDF)", data=pdf_bytes, file_name="Parecer_Auditoria_AutoMEP.pdf", type="primary")
            
        elif len(df_limpo) == 0:
            st.info("👈 Projeto aprovado geometricamente. Nenhuma auditoria de risco necessária.")
