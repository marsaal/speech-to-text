
# Com avaluar els models

Aquest repositori també inclou scripts per avaluar els models entrenats, tant qualitativament com quantitativament. Informació específica de què retorna cada mètode d'avaluació, juntament amb explicacions detallades de com executar i personalitzar els scripts es troba a continuació.

## Què retorna cada tipus d'script

### Anàlisi qualitatiu
Donat un conjunt de mostres (per defecte, 10), aquest script imprimeix per pantalla, per cada mostra:
* La transcripció de l'àudio, tal com està al dataset.
* La predicció del model, a partir de la mostra d'àudio.
* El valor de les mètriques WER i CER obtingudes, comparant la predicció del model amb la transcripció de l'àudio. Per més informació sobre aquestes mètriques, consultar [aquest fitxer](resultats.md#mètriques-utilitzades)

### Anàlisi quantitatiu
Aquest script imprimeix per pantalla la mitjana de WER i CER obtinguda per un model entrenat al fer les prediccions sobre tot el conjunt de test.

## Anàlisi quantitativa per accent (només per Deep Speech 2)
Aquest script imprimeix per pantalla la mitjana de WER i CER obtinguda per un model entrenat al fer les prediccions sobre tot el conjunt de test, separant els resultats pel tipus d'accent.

## Execució i personalització dels fitxers de test per model

### Model Baseline i Attention
* **Execució**: `python -m avaluacions.recurrents.(qualitatiu|quantitatiu)`
* **Personalització del nombre de mostres a mostrar al test qualitatiu**:
  1. Entrar al fitxer `avaluacions/recurrents/qualitatiu`
  2. Editar l'última línia, concretament, passar-li a la funció `demo_evaluate_sample` una llista amb els índexs de les mostres que es volen avaluar qualitativament.
* **Tria del model**:
  1. Entrar al fitxer `avaluacions/recurrents/qualitatiu` o `avaluacions/recurrents/quantitatiu`
  2. Editar la segona línia de la funció (`demo_evaluate_sample`), és a dir, posar a la variable `checkpoint_path`, la ruta al fitxer `.pth` on es troba el model entrenat
  3. Més endavant, on es crea el model, canviar la classe `Baseline` o `AttentionModel`, segons a quina classe pertanyi el model importat.
 
### Models basats en CTC
* **Execució**: `python -m avaluacions.ctc.(qualitatiu|quantitatiu)`
* **Personalització del nombre de mostres a mostrar al test qualitatiu**:
  1. Entrar al fitxer `avaluacions/ctc/qualitatiu`
  2. Editar l'última línia, concretament, passar-li a la funció `demo_evaluate_sample` una llista amb els índexs de les mostres que es volen avaluar qualitativament.
* **Tria del model**:
  1. Entrar al fitxer `avaluacions/ctc/qualitatiu` o `avaluacions/ctc/quantitatiu`
  2. Editar la segona línia de la funció (`demo_evaluate_sample`), és a dir, posar a la variable `checkpoint_path`, la ruta al fitxer `.pth` on es troba el model entrenat
  3. Més endavant, on es crea el model, canviar la classe `DeepCTCModel` o `DeepCTCModelV2`, segons a quina classe pertanyi el model importat.
 
### Model Deep Speech 2
* **Execució**: `python -m avaluacions.deep_speech2.(qualitatiu|quantitatiu|per_accent)`
* **Personalització del nombre de mostres a mostrar al test qualitatiu**:
  1. Entrar al fitxer `avaluacions/deep_speech2/qualitatiu`
  2. Editar la penúltima línia del fitxer, concretament, canviar el valor de la variable `idx` a una llista amb els índexs de les mostres que es volen avaluar qualitativament.
* **Tria del model**:
  1. Entrar al fitxer `avaluacions/deep_speech2/avaluacions`
  2. Editar la segona línia de la funció (`demo_evaluate_sample`), és a dir, posar a la variable `checkpoint_path`, la ruta al fitxer `.pth` on es troba el model entrenat
