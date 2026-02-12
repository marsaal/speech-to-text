
# Arquitectures proposades
## 1. Model Baseline
Model molt simple del qual es partia per provar que el nostre codi començava a fer prediccions (de poca qualitat).
### Inicialització del model
A l'hora d'inicialitzar el model cal donar valor als següents paràmetres:
* *input_dim*: Dimensió de les característiques d’àudio
* *hidden_dim*: Dimensió interna de les capes ocultes
* *embedding_dim*: Dimensió dels vectors d’embedding
* *num_classes*: Nombre total de classes de sortida (tokens)
### Arquitectura
Pel que fa a l'arquitectura, aquest primer model consta de les següents capes.
* *feature_extractor*: CNN unidimensional per extreure característiques de l'àudio.
* *encoder*: GRU unidireccional per codificar la seqüència d’àudio.
* *embedding*: capa per passar els tokens previs de dimensió *num_classes* a *embedding_dim*.
* *decoder*: GRU que combina l’embedding dels tokens previs i el context de l’encoder per generar la sortida.
* *classifier*: Capa lineal per predir la classe (token) a cada pas temporal.
### Forward pass
Donada una entrada, els tokens reals i la probabilitat amb què s'aplicarà teacher forcing, el foward pass del model és sel següent.
* Extracció de característiques sobre la mostra d'entrada.
* Es passa la seqüència processada per l'encoder per obtenir representacions temporals (encoder_outputs) i un estat final (encoder_hidden).
* Inicialització del decoder amb el token START i com a estat inicial s'utilitza l'estat final de l'encoder.

Aquí comença el bucle de decodificació, que consta dels següents passos (per cada pas temporal):
* Es transforma el token actual a embedding i es concatena amb el "context" (últim vector de l'encoder).
* Es passa pel decoder i es genera una predicció de token amb la capa classifier.
* S’utilitza teacher forcing o la predicció com a propera entrada (segons la probabilitat definida).

Finalment el model retorna la seqüència de prediccions (logits) per a cada pas de temps com a sortida.

### Problemes del model
Alguns problemes detectats en aquest primer model són els següents:
* El feature extraction es molt pobre ja que només té una capa convolucional i no és suficient per capturar els patrons.
* A l'hora de predir els caràcters no utilitza tota la informacio de la que disposa ja que nomes utilitza l'ultim output de l'encoder.

## 2. Model AttentionModel
Per tal de intentar solucionar els problemes anteriors, s'ha proposat un nou model basat en l'anterior. L'inicialització és la mateixa però l'arquitectura ha patit alguns canvis.
### Arquitectura
* *feature_extractor*: Ha passat de ser una sola convolució a 2 capes amb BatchNorm i Relu.
* *attention*: Component que calcula pesos per ressaltar les parts rellevants de la seqüència d’encoder segons l’estat actual del decoder
* *decoder*:Ara s'aplica l'atenció calculada en el context

Aquest esquema millora la capacitat del model per centrar-se en parts importants de l’input durant la generació, que era una de les mancances en el model Baseline. Tambe s'ha realitzat una extraccio de caracterstiques mes sofisticada. Tot i aixo, aquest model segueix tenint algunes mancances
### Problemes del model
* L'arquitectura actual és sequencial, de manera que es necessita un alineament explícit i a l'hora de processar els tokens es fa pas a pas. Això podria no ser el millor enfocament en una tasca de reconeixement de veu.
* L'atenció hauria d'anar sobre l'àudio.

## 3. Model DeepCTCModel
S'ha proposat un model amb un enfoc diferent a l'utilitzat en els anteriors. Ara s'utilizatà un model basat en CTC (Connectionist Temporal Classification). La inicialització segueix essent la mateixa però l'arquitecura ha patit grans canvis:
### Arquitectura
Aquest model consta de les següents capes principals:
* *feature_extractor*: Bloc de convolucions 1D amb batch normalization i ReLU. Permet extreure patrons temporals més complexos que el model baseline.
* *encoder*: GRU profund i bidireccional que processa la seqüència d’àudio i captura informació del passat i del futur simultàniament.
* *classifier*: Mòdul lineal amb ReLU i Dropout per classificar cada pas temporal a una de les classes de sortida.
### Forward pass
Donada una entrada d’àudio, el procés en aquest enfocament és diferent:
* Extracció de característiques utilitzant el bloc convolucional, obtenint una representació temporal millorada.
* La seqüència es passa per l'encoder, obtenint sortides per a cada pas temporal.
* Cada vector de la seqüència es transforma en logits (probabilitats no normalitzades) per a cada classe possible.

El model retorna una seqüència de prediccions (logits) per a cada pas de temps.
### Problemes del model
Aquest primer enfocament CTC és bastant simple i s'han detectat alguns problemes:
* Extracció de característiques massa simple
* L'encoder utilitzat té poca profunditat i capacitat de modelatge seqüencial
* La capa de classificació té poc poder de discriminació i robustesa davant el soroll.
* El model pot tenir dificultats per aprendre en profunditat per culpa del degradat del senyal

## 4. Model DeepCTCModelV2
Aquest nou model té una arquitectura més profunda que l'anterior i intenta solucionar alguns dels problemes del primer model CTC proposat. La inicialització i el forward pas es mantenen igual, però l'arquitura ha patit molts canvis.
### Arquitectura
Aquest model presenta una arquitectura molt més robusta i profunda
* *feature_extractor*: Ara està compost per diversos Residual Blocks, que milloren el flux de gradients i permeten xarxes profundes sense degradació. També s'inclou BatchNorm, ReLU i Dropout (per afavorir l'estabilitat i la regularització) i les capes convolucionals tenen diferents profunditats i mides de filtre.
* *encoder*: Ara és un bloc compost per 3 capes GRU bidireccionals, on cada una rep com a entrada la sortida de l’anterior.
* *classifier*: Ha passat a ser una xarxa amb 3 capes lineals. Inclou activacions ReLU i Dropout i redueix progressivament la dimensionalitat fins a arribar a la classificació final.

## 5. Model basat en el DeepSpeech2
Finalment, s'ha proposat un últim model de reconeixement automàtic, el qual s'ha inspirat en l'arquitectura de [DeepSpeech 2](https://github.com/sooftware/deepspeech2). Fa ús de convolucions 2D residuals, normalització capa a capa i una pila de GRUs bidireccionals.

### Inicialització del model
A l'hora d'inicialitzar el model cal definir un seguit de paràmetres:
* *n_cnn_layers*: Nombre de capes convolucionals residuals.
* *n_rnn_layers*: Nombre de capes GRU bidireccionals.
* *rnn_dim*: Dimensió de sortida de les GRUs.
* *n_class*: Nombre total de classes de sortida (tokens).
* *n_feats*: Nombre de característiques d’entrada (ex: MFCCs o spectrograma).
* *stride*: Pas utilitzat a la primera convolució.
* *dropout*: Taxa de desactivació per regularització.

### Arquitectura
Aquest últim model es divideix en diferents blocs:

* CNN inicial: Una capa Conv2D amb que redueix la resolució temporal de l’entrada i extreu característiques jeràrquiques.
* Bloc convolucional residual: Conjunt de ResidualCNN (n capes), cadascuna formada per dues convolucions amb GELU, Dropout i LayerNorm (en lloc de BatchNorm, millor per seqüències) i connexió residual que afegeix l'entrada original al final del bloc.
* Fully Connected: Conversió de la sortida de la CNN a una representació vectorial per ser introduïda al bloc recurrent.
* Bloc recurrent: Una seqüència de GRUs bidireccionals, cada una processant la seqüència amb LayerNorm i GELU abans de la GRU i Dropout després de la GRU.
* Classificador: Xarxa de classificació amb dues capes lineals