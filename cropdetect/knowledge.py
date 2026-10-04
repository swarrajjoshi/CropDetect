"""Versioned curated knowledge base. Educational guidance, not a pesticide registry."""
from cropdetect.config import REVIEW_THRESHOLD

KB_VERSION = '2026-09-21'
UMN = 'https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/tomato-leaf-spot-diseases'
KB = {
 'early_blight': {
  'kind': 'fungal', 'treatment': ['Remove affected lower leaves conservatively and sanitize pruning tools.', 'Use cultural controls first; obtain local advice before considering a fungicide.'],
  'prevention': ['Water at soil level and mulch to limit splash.', 'Stake plants, allow airflow, and rotate away from tomatoes and peppers for two years.'],
  'sources': [{'title': 'UMN Extension: Early blight', 'url': 'https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/early-blight-in-tomato-and-potato'}]},
 'septoria': {
  'kind': 'fungal', 'treatment': ['Remove infected leaves and clear infected residue at season end.', 'For recurring severe disease, extension guidance discusses labelled fungicides including fixed copper; consult local registration and an agronomist before use.'],
  'prevention': ['Use mulch and irrigation at the base to reduce splash and leaf wetness.', 'Space plants for airflow and inspect transplants before planting.'],
  'sources': [{'title': 'University of Maryland Extension: Septoria', 'url': 'https://www.extension.umd.edu/resource/septoria-leaf-spot-tomatoes'}]},
 'spider_mites': {
  'kind': 'pest', 'treatment': ['Confirm live mites or webbing on leaf undersides with a hand lens before treatment.', 'Water sprays can reduce mites. Extension guidance lists insecticidal soaps or horticultural oils when needed; verify the tomato label and local authorization. Avoid application in heat or to water-stressed plants.'],
  'prevention': ['Avoid drought stress and reduce dust.', 'Protect predatory mites and other natural enemies; avoid unnecessary broad-spectrum insecticides.'],
  'sources': [{'title': 'UC IPM: Spider mites', 'url': 'https://ipm.ucanr.edu/home-and-landscape/spider-mites/'}]},
 'healthy': {
  'kind': 'healthy', 'treatment': ['No treatment is indicated by this image classification. Continue inspecting the plant.'],
  'prevention': ['Maintain airflow, mulch, and water at the soil.', 'Inspect leaves regularly; an apparently healthy image does not exclude early infection.'],
  'sources': [{'title': 'UMN Extension: Tomato leaf spot prevention', 'url': UMN}]}
}


def recommendation(label):
    return dict(KB[label], label=label, provider='Curated extension knowledge base', version=KB_VERSION,
                scope='US extension education; not an India pesticide registration service. No dosage is prescribed. Follow locally registered crop/pest labels and seek local expert confirmation.')


def reason(label, confidence, pest_probability, coverage, threshold=REVIEW_THRESHOLD):
    """Forward-chaining rules over explicitly recorded facts; abstention dominates."""
    facts = {'candidate': label, 'confidence': confidence, 'pest_probability': pest_probability,
             'usable_mask': .08 <= coverage <= .95, 'above_threshold': confidence >= threshold,
             'detectors_agree': (label == 'spider_mites') == (pest_probability >= .5)}
    trace = []
    for rule, passed, explanation in [
        ('R1', facts['usable_mask'], 'Leaf-mask coverage must be between 8% and 95%.'),
        ('R2', facts['above_threshold'], f'Candidate probability must reach the policy threshold {threshold:.0%}.'),
        ('R3', facts['detectors_agree'], 'Multiclass and binary pest decisions must agree.')]:
        trace.append({'rule': rule, 'passed': passed, 'explanation': explanation})
    accepted = all(t['passed'] for t in trace)
    trace.append({'rule': 'R4', 'passed': accepted, 'explanation': 'If R1-R3 pass, retrieve class guidance; otherwise request expert review and withhold class-specific treatment.'})
    return {'status': 'supported_candidate' if accepted else 'needs_review', 'facts': facts, 'trace': trace,
            'recommendation': recommendation(label) if accepted else {
                'provider': 'Uncertainty policy', 'treatment': ['Do not treat based on this uncertain result. Retake a clear image and seek an agronomist’s diagnosis.'],
                'prevention': ['Monitor the plant and inspect both leaf surfaces.'], 'sources': [],
                'scope': 'Out-of-scope images can still receive high probabilities. This is not a general plant or insect recognizer.'}}
