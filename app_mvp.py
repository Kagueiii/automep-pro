import streamlit as st
import fitz  
import tempfile
import os
import pandas as pd
import google.generativeai as genai
from core_engine import IngestorPDFLocal, TriagemEspacialLocal

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

# 3. CABEÇALHO
st.markdown('<p class="main-title">🏗️ AutoMEP Pro <sub>v1.0</sub></p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Motor de Triagem Espacial e Detecção de Interferências (Clash Detection)</p>', unsafe_allow_html=True)

# 4. BARRA LATERAL
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=60)
    st.header("Upload de Prancha")
    arquivo_pdf = st.file_uploader("Formato suportado: .PDF", type=["pdf"])
    
    st.markdown("---")
    st.subheader("⚙️ Calibragem do Motor")
    sensibilidade = st.slider("Tolerância de Proximidade (px)", 1.0, 20.0, 5.0, 1.0)
    
    st.markdown("---")
    st.subheader("🧠 Cérebro IA (Gemini)")
    st.caption("Insira sua chave para ativar o Laudo Automático")
    chave_api = st.text_input("API Key do Google", type="password")

# 5. LÓGICA PRINCIPAL
if arquivo_pdf is not None:
    with st.spinner("⏳ Triangulando coordenadas espaciais..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(arquivo_pdf.read())
            caminho_tmp = tmp.name

        try:
            # Roda o Motor
            ingestor = IngestorPDFLocal()
            componentes, w, h = ingestor.processar_pdf(caminho_tmp)
            motor = TriagemEspacialLocal(limite_conflito_px=sensibilidade)
            clusters = motor.executar_triagem(componentes, w, h)

            # Renderiza Quadrados Vermelhos
            doc = fitz.open(caminho_tmp)
            pagina = doc.load_page(0)
            for cluster in clusters:
                for id_comp in cluster:
                    comp = motor.mapa_componentes[id_comp]
                    b = comp.bbox_relativo
                    rect = fitz.Rect(b[0]*w, b[1]*h, b[2]*w, b[3]*h)
                    rect.x0 -= 3; rect.y0 -= 3; rect.x1 += 3; rect.y1 += 3
                    pagina.draw_rect(rect, color=(1, 0, 0), width=2, fill_opacity=0.3, fill=(1, 1, 0))
            
            pix = pagina.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            doc.close() # Trava de segurança do Windows

            # Tabela de Dados
            dados_tabela = []
            for i, cluster in enumerate(clusters):
                sistemas_env = list(set([motor.mapa_componentes[id_c].tipo_sistema for id_c in cluster]))
                elementos = [motor.mapa_componentes[id_c].texto_extraido for id_c in cluster]
                dados_tabela.append({
                    "ID do Clash": f"CLASH-{i+1:03d}",
                    "Sistemas": " / ".join(sistemas_env).upper(),
                    "Elementos Descritos": " | ".join([e for e in elementos if e != "[GEOMETRIA VETORIAL]"]),
                    "Gravidade": "🔴 Alta", "Status": "Pendente"
                })
            df_conflitos = pd.DataFrame(dados_tabela)

            # Dashboard - Topo
            if len(clusters) == 0:
                st.success("✅ PROJETO APROVADO: Nenhuma interferência detectada.")
            else:
                st.error(f"⚠️ ATENÇÃO: Foram detectadas {len(clusters)} interferências críticas.")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Componentes Lidos", len(componentes))
            m2.metric("Tamanho da Prancha", f"{int(w)}x{int(h)}")
            m3.metric("Hard Clashes", len(clusters))
            m4.metric("Índice de Risco", f"{min(100, len(clusters)*15)}%")
            st.markdown("<br>", unsafe_allow_html=True)

            # Abas
            tab1, tab2, tab3 = st.tabs(["🗺️ Mapa Espacial", "📑 Relatório de Log", "🤖 Laudo da IA"])
            
            with tab1:
                st.image(img_bytes, use_container_width=True)

            with tab2:
                if not df_conflitos.empty:
                    st.dataframe(df_conflitos, use_container_width=True, hide_index=True)
                    csv = df_conflitos.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Exportar (.CSV)", data=csv, file_name='conflitos.csv', mime='text/csv')

            with tab3:
                st.subheader("🧠 Assistente Técnico de Engenharia")
                if len(clusters) > 0:
                    if not chave_api:
                        st.warning("⚠️ Insira sua API Key do Google na barra lateral para liberar a geração de laudos.")
                    
                    # Botão Mágico
                    # Botão Mágico
                    if st.button("✨ Gerar Laudo Automático", disabled=not chave_api):
                        genai.configure(api_key=chave_api)
                        
                        # --- INÍCIO DO CÓDIGO AUTO-DESCOBRIDOR ---
                        modelo_escolhido = "gemini-pro" # Fallback de segurança
                        try:
                            # Pergunta ao Google quais modelos estão liberados para você
                            for m in genai.list_models():
                                if 'generateContent' in m.supported_generation_methods:
                                    modelo_escolhido = m.name
                                    break # Pega o primeiro modelo válido e foge!
                        except Exception as e:
                            st.error(f"Erro ao listar modelos do Google: {e}")
                            
                        # Usa o modelo que o próprio Google recomendou
                        modelo_ia = genai.GenerativeModel(modelo_escolhido)
                        # --- FIM DO CÓDIGO AUTO-DESCOBRIDOR ---
                        
                        for index, linha in df_conflitos.iterrows():
                            with st.expander(f"Análise: {linha['ID do Clash']}", expanded=True):
                                with st.spinner(f"IA Redigindo Laudo via {modelo_escolhido}..."):
                                    prompt_engenharia = f"""
                                    Você é um Engenheiro Sênior de compatibilização de projetos (BIM/MEP).
                                    O motor espacial detectou um Hard Clash (Colisão física) entre os seguintes sistemas:
                                    - Sistemas: {linha['Sistemas']}
                                    - Elementos identificados: {linha['Elementos Descritos']}
                                    
                                    Escreva um laudo técnico curto, direto e profissional contendo:
                                    1. Risco Principal
                                    2. Análise do Conflito
                                    3. Solução Sugerida para a obra.
                                    Use formatação em Markdown (negrito, tópicos).
                                    Vá direto ao ponto. Não crie cabeçalhos, não invente datas, IDs ou nomes de projetos.
                                    """
                                    try:
                                        resposta = modelo_ia.generate_content(prompt_engenharia)
                                        st.markdown(resposta.text)
                                    except Exception as e:
                                        st.error(f"Erro na comunicação com a IA: {e}")
                else:
                    st.success("Sem alertas para a IA analisar.")

        finally:
            os.remove(caminho_tmp) 
else:
    st.info("👈 Envie sua prancha na barra lateral para começar.")