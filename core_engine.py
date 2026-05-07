import fitz  
import math
import logging
import re
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
        
        # Extração de Texto
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

        # Extração de Vetores
        desenhos = pagina.get_drawings()
        for d in desenhos:
            rx0, ry0, rx1, ry1 = d["rect"]
            if (rx1 - rx0) < 1 and (ry1 - ry0) < 1: continue
            if (rx1 - rx0) > w_nativo * 0.95: continue
            
            tipo_vetor = "eletrica" 
            componentes.append(ComponenteMEP(
                id_unico=f"vetor_{contador}",
                tipo_sistema=tipo_vetor,
                bbox_relativo=[rx0/w_nativo, ry0/h_nativo, rx1/w_nativo, ry1/h_nativo],
                texto_extraido="[GEOMETRIA VETORIAL]"
            ))
            contador += 1
                            
        doc.close()
        return componentes, w_nativo, h_nativo

    def _classificar_tipo_mock(self, texto: str) -> str:
        texto_lower = texto.lower()
        # Vocabulário refinado: 'mm' atrelado à elétrica (bitola)
        if any(kw in texto_lower for kw in ["tug", "tue", "qdc", "qd", "disjuntor", " a ", " v ", " w ", "mm", "potência", "circuito", "neutro", "fase", "terra"]):
            return "eletrica"
        elif any(kw in texto_lower for kw in ["af ", "aq ", "esgoto", "pvc", "cpvc", "ø", "chuveiro", "ralo", "registro", "caixa d'água"]):
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
        self.grafo.clear()
        self.idx_rtree = index.Index(properties=index.Property())
        self.mapa_componentes.clear()

        for i, ent in enumerate(entidades):
            self.mapa_componentes[i] = ent
            self.idx_rtree.insert(i, ent.bbox_relativo)

        nos_suspeitos = set()
        for id_atual, comp_atual in self.mapa_componentes.items():
            b = comp_atual.bbox_relativo
            bbox_busca = (b[0]-0.01, b[1]-0.01, b[2]+0.01, b[3]+0.01)
            for id_vizinho in self.idx_rtree.intersection(bbox_busca):
                if id_vizinho != id_atual:
                    comp_v = self.mapa_componentes[id_vizinho]
                    dist = self._distancia_px(comp_atual.bbox_relativo, comp_v.bbox_relativo, w, h)
                    if comp_atual.tipo_sistema != comp_v.tipo_sistema and dist <= self.limite_conflito_px:
                        nos_suspeitos.add(id_atual)
                        nos_suspeitos.add(id_vizinho)
                        self.grafo.add_edge(id_atual, id_vizinho)
                        
        clusters_iniciais = [list(c) for c in nx.connected_components(self.grafo) if nos_suspeitos.intersection(c)]
        
        # Filtro Enterprise: Apenas colisões entre sistemas reais distintos
        clusters_validados = []
        for cluster in clusters_iniciais:
            sistemas_presentes = set([self.mapa_componentes[id_c].tipo_sistema for id_c in cluster])
            sistemas_reais = {s for s in sistemas_presentes if s not in ["arquitetura_generica", "INDEFINIDO", "TEXTO"]}
            if len(sistemas_reais) >= 2:
                clusters_validados.append(cluster)
        return clusters_validados
