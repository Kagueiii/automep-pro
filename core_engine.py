import fitz
import math
import logging
import re
from typing import List, Dict, Tuple, Optional, Any, Set
from rtree import index
import networkx as nx
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ComponenteMEP(BaseModel):
    id_unico: str
    tipo_sistema: str
    bbox_relativo: List[float]
    texto_extraido: str
    status_auditoria: str = "Crítico"

class IngestorPDFLocal:
    def _normalizar_cor(self, cor) -> Tuple[float, float, float]:
        if not cor or not isinstance(cor, (list, tuple)):
            return (0.0, 0.0, 0.0)
        if len(cor) == 3: return tuple(cor)
        if len(cor) == 4:
            c, m, y, k = cor
            return ((1.0-c)*(1.0-k), (1.0-m)*(1.0-k), (1.0-y)*(1.0-k))
        if len(cor) == 1: return (cor[0], cor[0], cor[0])
        return (0.0, 0.0, 0.0)

    def processar_pdf(self, caminho_pdf: str, pagina_alvo: int = 0, disciplina_mestra: Optional[str] = None) -> Tuple[List[ComponenteMEP], float, float]:
        doc = fitz.open(caminho_pdf)
        try:
            if pagina_alvo >= len(doc): pagina_alvo = 0
            pagina = doc.load_page(pagina_alvo)
            pagina.set_rotation(0)
            pagina.set_cropbox(pagina.mediabox)
            w_nativo, h_nativo = pagina.rect.width, pagina.rect.height
            componentes, contador = [], 0
            
            blocos = pagina.get_text("dict").get("blocks", [])
            for b in blocos:
                if "lines" in b:
                    for l in b["lines"]:
                        for span in l["spans"]:
                            texto = span["text"].strip()
                            if texto:
                                tipo_base = self._classificar_tipo_mock(texto)
                                tipo = disciplina_mestra if (disciplina_mestra and tipo_base != "arquitetura_generica") else tipo_base
                                x0, y0, x1, y1 = span["bbox"]
                                componentes.append(ComponenteMEP(
                                    id_unico=f"txt_{contador}", tipo_sistema=tipo,
                                    bbox_relativo=[round(x0/w_nativo, 5), round(y0/h_nativo, 5), round(x1/w_nativo, 5), round(y1/h_nativo, 5)],
                                    texto_extraido=texto
                                ))
                                contador += 1

            desenhos = pagina.get_drawings()
            if desenhos:
                step = max(1, len(desenhos) // 150)
                amostra = sorted([d.get("width", 0.1) for d in desenhos[::step]])
                media_tracos = sum(amostra[1:-1]) / len(amostra[1:-1]) if len(amostra) > 2 else 1.0
            else: media_tracos = 1.0

            for d in desenhos[:15000]:
                rx0, ry0, rx1, ry1 = d["rect"]
                if (rx1 - rx0) < 1.0 and (ry1 - ry0) < 1.0: continue
                cor_rgb = self._normalizar_cor(d.get("color"))
                largura, w_bb, h_bb = d.get("width", 0.1), rx1 - rx0, ry1 - ry0
                ratio = max(w_bb, h_bb) / max(min(w_bb, h_bb), 0.001)
                
                if disciplina_mestra: tipo_vetor = disciplina_mestra
                else:
                    tipo_vetor = "eletrica"
                    if ratio > 10 and cor_rgb[2] > cor_rgb[0] and largura > (media_tracos * 2.5):
                        tipo_vetor = "hidraulica"
                componentes.append(ComponenteMEP(
                    id_unico=f"vetor_{contador}", tipo_sistema=tipo_vetor,
                    bbox_relativo=[round(rx0/w_nativo, 5), round(ry0/h_nativo, 5), round(rx1/w_nativo, 5), round(ry1/h_nativo, 5)],
                    texto_extraido=f"[INFRA {tipo_vetor.upper()} R={ratio:.1f} W={largura:.2f}]"
                ))
                contador += 1
            return componentes, w_nativo, h_nativo
        finally: doc.close()

    def _classificar_tipo_mock(self, texto: str) -> str:
        t = texto.lower()
        if "caixa" in t:
            if any(k in t for k in ["passagem", "4x2", "4x4", "octogonal"]): return "eletrica"
            if any(k in t for k in ["agua", "esgoto", "inspecao", "gordura"]): return "hidraulica"
        if "mm" in t:
            if any(k in t for k in ["pvc", "af", "aq", "esgoto", "tubo"]): return "hidraulica"
            return "eletrica"
        if any(k in t for k in ["tug", "tue", "qdc", "disjuntor", "quadro", "fio", "circ", "c."]): return "eletrica"
        if any(k in t for k in ["af", "aq", "esgoto", "pluvial", "tubo", "ralo"]): return "hidraulica"
        return "arquitetura_generica"

class TriagemEspacialLocal:
    def __init__(self, limite_conflito_px: float = 1.0):
        self.limite_conflito_px = limite_conflito_px
        self.idx_rtree = index.Index(properties=index.Property())
        self.grafo = nx.Graph()
        self.mapa_componentes: Dict[int, ComponenteMEP] = {}

    def _extrair_valor_metadado(self, texto: str, chave: str) -> float:
        match = re.search(rf'{chave}=(\d+[.,]\d+|\d+)', texto)
        if match:
            try: return float(match.group(1).replace(',', '.'))
            except: return 1.0
        return 1.0

    def _distancia_px(self, b1, b2, w, h):
        dx = max(0.0, b2[0] - b1[2], b1[0] - b2[2])
        dy = max(0.0, b2[1] - b1[3], b1[1] - b2[3])
        return math.sqrt((dx * w)**2 + (dy * h)**2)

    def _obter_elevacao(self, texto: str) -> Optional[float]:
        match = re.search(r'(?:[+-]|[hH]=)\s*(\d+[.,]?\d*)\s*(m|cm)?', texto, re.I)
        if match:
            try:
                val = float(match.group(1).replace(',', '.'))
                if (match.group(2) or "").lower() == "cm": val /= 100.0
                return val
            except: return None
        return None

    def executar_triagem(self, entidades: List[ComponenteMEP], w: float, h: float, disciplina_mestra: Optional[str] = None) -> List[Dict[str, Any]]:
        self.grafo.clear()
        self.idx_rtree = index.Index(properties=index.Property())
        self.mapa_componentes.clear()
        densidade_global = len(entidades) / 1.0
        limite_efetivo = 0.5 if disciplina_mestra else self.limite_conflito_px
        for i, ent in enumerate(entidades):
            if ent.tipo_sistema == "arquitetura_generica": continue
            self.mapa_componentes[i] = ent
            self.idx_rtree.insert(i, ent.bbox_relativo)

        nos_suspeitos = set()
        for id_atual, comp_atual in self.mapa_componentes.items():
            b = comp_atual.bbox_relativo
            bbox_busca = (b[0]-0.005, b[1]-0.005, b[2]+0.005, b[3]+0.005)
            vizinhos = list(self.idx_rtree.intersection(bbox_busca))
            densidade_local = len(vizinhos) / 0.0001
            limite_local = limite_efetivo * 0.7 if densidade_local > (3.0 * densidade_global) else limite_efetivo
            for id_vizinho in vizinhos:
                if id_vizinho != id_atual:
                    comp_v = self.mapa_componentes[id_vizinho]
                    if disciplina_mestra:
                        ratio_v = self._extrair_valor_metadado(comp_v.texto_extraido, "R")
                        if comp_atual.id_unico.startswith("txt") and ratio_v > 10: continue
                    if self._distancia_px(comp_atual.bbox_relativo, comp_v.bbox_relativo, w, h) <= limite_local:
                        if disciplina_mestra or comp_atual.tipo_sistema != comp_v.tipo_sistema:
                            nos_suspeitos.update([id_atual, id_vizinho])
                            self.grafo.add_edge(id_atual, id_vizinho)
                        
        clusters_brutos = [list(c) for c in nx.connected_components(self.grafo) if nos_suspeitos.intersection(c)]
        clusters_consolidados = []
        regex_circ = r'\b(c|circ|cir|circuito)[.-]?\s*\d+\b'

        for cluster in clusters_brutos:
            c_x0, c_y0, c_x1, c_y1 = 1.0, 1.0, 0.0, 0.0
            textos, circs, maior_w, tem_txt = [], set(), 0.0, False
            for n in cluster:
                comp = self.mapa_componentes[n]
                bx0, by0, bx1, by1 = comp.bbox_relativo
                c_x0, c_y0, c_x1, c_y1 = min(c_x0, bx0), min(c_y0, by0), max(c_x1, bx1), max(c_y1, by1)
                textos.append(comp.texto_extraido)
                if comp.id_unico.startswith("txt"): tem_txt = True
                w_val = self._extrair_valor_metadado(comp.texto_extraido, "W")
                if w_val > maior_w: maior_w = w_val
                m = re.search(regex_circ, comp.texto_extraido, re.I)
                if m: circs.add(m.group(0).upper())

            area = max(1e-7, (c_x1-c_x0)*(c_y1-c_y0))
            if (len(cluster)/area) > 3.0 * densidade_global:
                if (len(set([t.lower() for t in textos])) < len(textos)*0.4 or "legenda" in " ".join(textos).lower()) and len(circs) <= 1: continue

            raio_fisico_rel = 40 / w
            elevs = []
            for n in cluster:
                bx0, by0, bx1, by1 = self.mapa_componentes[n].bbox_relativo
                bbox_loc = (bx0 - raio_fisico_rel, by0 - raio_fisico_rel, bx1 + raio_fisico_rel, by1 + raio_fisico_rel)
                for v_id in self.idx_rtree.intersection(bbox_loc):
                    h_val = self._obter_elevacao(self.mapa_componentes[v_id].texto_extraido)
                    if h_val is not None: elevs.append(h_val)

            status = "Cruzamento em Níveis Diferentes" if len(set(elevs)) > 1 else "Crítico"
            categoria = "OVERLAP: Legibilidade" if tem_txt else "HARD CLASH: Impedimento Físico"
            desc = f"Infra principal (W={maior_w:.2f})"
            if circs: desc += f" | Circuitos: {', '.join(circs)}"
            desc += f" | Detalhes: {' | '.join([t for t in textos if '[' not in t])}"

            clusters_consolidados.append({"ids": cluster, "status": status, "categoria_conflito": categoria, "descricao_tecnica": desc})
                
        return clusters_consolidados

# AutoMEP Pro v1.0 - LTS Production Build
