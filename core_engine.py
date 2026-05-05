import fitz  
import math
import logging
from typing import List, Dict, Tuple
from rtree import index
import networkx as nx
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ComponenteMEP(BaseModel):
    id_unico: str
    tipo_sistema: str
    bbox_relativo: List[float]
    texto_extraido: str

class IngestorPDFLocal:
    def processar_pdf(self, caminho_pdf: str) -> Tuple[List[ComponenteMEP], float, float]:
        doc = fitz.open(caminho_pdf)
        pagina = doc.load_page(0)
        pagina.set_rotation(0)
        pagina.set_cropbox(pagina.mediabox)
        
        w_nativo = pagina.rect.width
        h_nativo = pagina.rect.height
        
        componentes = []
        contador = 0
        
        # Extração de Texto (Identificação de Tomadas, Quadros, etc)
        blocos = pagina.get_text("dict").get("blocks", [])
        for b in blocos:
            if "lines" in b:
                for l in b["lines"]:
                    for span in l["spans"]:
                        texto = span["text"].strip()
                        if texto:
                            tipo = self._classificar_tipo_mock(texto)
                            x0, y0, x1, y1 = span["bbox"]
                            componentes.append(ComponenteMEP(
                                id_unico=f"txt_{contador}",
                                tipo_sistema=tipo,
                                bbox_relativo=[x0/w_nativo, y0/h_nativo, x1/w_nativo, y1/h_nativo],
                                texto_extraido=texto
                            ))
                            contador += 1

        # Extração de Vetores (Identificação de Tubos e Linhas)
        desenhos = pagina.get_drawings()
        for d in desenhos:
            rx0, ry0, rx1, ry1 = d["rect"]
            
            # Filtro Matemático: Só ignora se for literalmente um ponto invisível
            if (rx1 - rx0) < 1 and (ry1 - ry0) < 1:
                continue
            # Ignora linhas gigantes que formam a margem do PDF
            if (rx1 - rx0) > w_nativo * 0.95:
                continue
                
            cor = d.get("color")
            tipo_vetor = "eletrica" # Assume elétrica por padrão
            
            # Filtro de Cores: Se tiver mais azul que vermelho no RGB, é Hidráulica
            if isinstance(cor, (list, tuple)) and len(cor) >= 3:
                if cor[2] > cor[0]: 
                    tipo_vetor = "hidraulica"
                    
            componentes.append(ComponenteMEP(
                id_unico=f"vetor_{contador}",
                tipo_sistema=tipo_vetor,
                bbox_relativo=[rx0/w_nativo, ry0/h_nativo, rx1/w_nativo, ry1/h_nativo],
                texto_extraido="[GEOMETRIA VETORIAL]"
            ))
            contador += 1
                            
        doc.close()
        # RETORNO - Exatamente alinhado com o escopo da função
        return componentes, w_nativo, h_nativo

    def _classificar_tipo_mock(self, texto: str) -> str:
        texto_lower = texto.lower()
        if any(kw in texto_lower for kw in ["tug", "tue", "qdc", "disjuntor", "a", "v", "w"]):
            return "eletrica"
        elif any(kw in texto_lower for kw in ["af", "aq", "esgoto", "tubo", "mm"]):
            return "hidraulica"
        return "arquitetura_generica"

class TriagemEspacialLocal:
    def __init__(self, limite_conflito_px: float = 5.0):
        self.limite_conflito_px = limite_conflito_px
        self.idx_rtree = index.Index(properties=index.Property())
        self.grafo = nx.Graph()
        self.mapa_componentes: Dict[int, ComponenteMEP] = {}

    def _distancia_px(self, b1, b2, w, h):
        dx = max(0.0, b2[0] - b1[2], b1[0] - b2[2])
        dy = max(0.0, b2[1] - b1[3], b1[1] - b2[3])
        return math.sqrt((dx * w)**2 + (dy * h)**2)

    def executar_triagem(self, entidades, w, h):
        # Limpa o estado anterior para evitar sobreposição de resultados no Streamlit
        self.grafo.clear()
        self.idx_rtree = index.Index(properties=index.Property())
        self.mapa_componentes.clear()

        # Alimenta o Índice Espacial (RTree)
        for i, ent in enumerate(entidades):
            self.mapa_componentes[i] = ent
            self.idx_rtree.insert(i, ent.bbox_relativo)

        nos_suspeitos = set()
        
        # Checa interferências cruzadas
        for id_atual, comp_atual in self.mapa_componentes.items():
            b = comp_atual.bbox_relativo
            bbox_busca = (b[0]-0.01, b[1]-0.01, b[2]+0.01, b[3]+0.01)
            
            for id_vizinho in self.idx_rtree.intersection(bbox_busca):
                if id_vizinho != id_atual: # Não pode colidir consigo mesmo
                    comp_v = self.mapa_componentes[id_vizinho]
                    dist = self._distancia_px(comp_atual.bbox_relativo, comp_v.bbox_relativo, w, h)
                    
                    # REGRA MESTRA: Sistemas de tipos diferentes e distâncias muito curtas
                    if comp_atual.tipo_sistema != comp_v.tipo_sistema and dist <= self.limite_conflito_px:
                        nos_suspeitos.add(id_atual)
                        nos_suspeitos.add(id_vizinho)
                        self.grafo.add_edge(id_atual, id_vizinho)
                        
        # Pega os agrupamentos brutos
        clusters_iniciais = [list(c) for c in nx.connected_components(self.grafo) if nos_suspeitos.intersection(c)]
        
        # --- INÍCIO DO FILTRO ENTERPRISE (REGRA DO BILHÃO) ---
        clusters_validados = []
        
        for cluster in clusters_iniciais:
            # Identifica todas as disciplinas (sistemas) presentes neste choque
            sistemas_presentes = set([self.mapa_componentes[id_c].tipo_sistema for id_c in cluster])
            
            # Removemos ruídos visuais genéricos
            sistemas_reais = {s for s in sistemas_presentes if s not in ["arquitetura_generica", "INDEFINIDO", "TEXTO", "GERAL"]}
            
            # A Regra de Ouro: Só consideramos um Hard Clash real se houver 2 ou mais disciplinas DIFERENTES envolvidas.
            if len(sistemas_reais) >= 2:
                clusters_validados.append(cluster)
                
        return clusters_validados
        # --- FIM DO FILTRO ENTERPRISE ---
