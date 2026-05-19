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
                # [CORREÇÃO 2] Limite de segurança: analisa no máximo os 15 erros mais graves para não estourar tokens
                for _, r in df_limpo.head(15).iterrows():
                    batch_str += f"- **{r['ID do Clash']}** ({r['Categoria']}) | Status Z: {r['Status Técnica']}\n  - Contexto: {r['Descrição Técnica']}\n"
                
                prompt_auditor = f"""Você é um Auditor Sênior MEP e Orçamentista Pessimista de Obras.
Analise esta lista de defeitos críticos purificados pelo nosso motor:

{batch_str}

Para CADA defeito, retorne:
1. Violação de Norma: Cite NBRs aplicáveis (ex: 5410, 8160, etc).
2. Falha Operacional: O que vai quebrar em 5 anos?
3. Custo de Retrabalho: Estime o pior cenário em R$ (material + mão de obra).

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

        # [CORREÇÃO 1] Proteção de UX: Garante que o texto não desapareça se a página recarregar
        if st.session_state["laudo_salvo"]:
            # Se o texto já estava salvo e NÃO foi gerado agora, imprime com markdown estático
            if not gerado_nesta_sessao:
                st.markdown(st.session_state["laudo_salvo"])
                
            pdf_bytes = gerar_pdf_bytes(st.session_state["laudo_salvo"])
            st.download_button("📥 Baixar Parecer Técnico (PDF)", data=pdf_bytes, file_name="Parecer_Auditoria_AutoMEP.pdf", type="primary")
            
        elif len(df_limpo) == 0:
            st.info("👈 Projeto aprovado geometricamente. Nenhuma auditoria de risco necessária.")
