# AI for VET – Pedagogicko-technická príručka pre učiteľov
## Integrácia interaktívnych simulácií umelej inteligencie a strojového učenia do Moodle

*Projekt: **AI for VET** (Artificial Intelligence for Vocational Education and Training)*  
*Program: **Erasmus+***  
*Oficiálny portál:* [https://aiforvet.eu/](https://aiforvet.eu/)  
*Simulácie bez Moodle:* [https://sim.aiforvet.eu/](https://sim.aiforvet.eu/)  
*Autori a technický tím projektu AI for VET*  
*Verzia príručky: 1.5 (november 2026)*  
*Kompatibilita: Moodle 4.5 LTS, 5.0, 5.1, 5.2, 5.3+*

---

## Obsah
1. [Úvod a pedagogický rámec](#1-úvod-a-pedagogický-rámec)
   - 1.1. Výzva: Prečo AI v odbornom vzdelávaní?
   - 1.2. Pedagogický prístup: Od vizuálnej intuície k inžinierskemu porozumeniu
   - 1.3. Architektúra balíka pre Moodle (filter, typ otázky, editor TinyMCE)
2. [Pedagogický sprievodca 13 simuláciami](#2-pedagogický-sprievodca-13-simuláciami)
   - 2.1. Modul 1: Gradientný zostup a optimalizácia modelu (`gradient_descent`)
   - 2.2. Modul 2: Matica zámen a rozhodovací prah (`confusion_matrix`)
   - 2.3. Modul 3: Polynomiálna regresia a regularizácia (`regression`)
   - 2.4. Modul 4: Nelineárne rozhodovacie hranice – k-NN a rozhodovací strom (`knn_tree`)
   - 2.5. Modul 5: Učenie bez učiteľa a zhlukovanie K-Means (`kmeans`)
   - 2.6. Modul 6: Perceptrón a neurónová sieť (`neural_network`)
   - 2.7. Modul 7: Konvolúcia a obrazové filtre (`convolution`)
   - 2.8. Modul 8: Posilňované učenie – Q-learning (`q_learning`)
   - 2.9. Modul 9: Rozdelenie dát a skreslenie (`data_bias`)
   - 2.10. Modul 10: Logistická regresia (`logistic`)
   - 2.11. Modul 11: Krížová validácia a hyperparametre (`cross_validation`)
   - 2.12. Modul 12: Príprava dát a EDA (`data_prep`)
   - 2.13. Modul 13: Rekurentná neurónová sieť (`rnn`)
3. [Technický návod pre učiteľov](#3-technický-návod-pre-učiteľov)
   - 3.1. Vkladanie simulácií vizuálnym editorom TinyMCE (`tiny_aisim`)
   - 3.2. Tvorba automaticky hodnotených úloh (`qtype_aisim`)
   - 3.3. Import hotovej banky otázok (Moodle XML)
   - 3.4. AISim Lab: simulácie bez Moodle
4. [Metodické odporúčania a organizácia vyučovania](#4-metodické-odporúčania-a-organizácia-vyučovania)
   - 4.1. Štruktúra 90-minútovej praktickej hodiny
   - 4.2. Formatívne hodnotenie a učenie z chýb
   - 4.3. Diferencované vyučovanie
   - 4.4. Odporúčané poradie simulácií
5. [Príloha a prehľad vzorcov](#5-príloha-a-prehľad-vzorcov)

---

## 1. Úvod a pedagogický rámec

### 1.1. Výzva: Prečo AI v odbornom vzdelávaní?
V mechatronike, elektrotechnike, priemyselnej automatizácii, zdravotníckej technike a IT už umelá inteligencia (AI) a strojové učenie (ML) nie sú len témou vysokoškolského výskumu. Žiaci odborných škôl (VET – *Vocational Education and Training*) sa dnes stretávajú s:
- automatizovanou optickou kontrolou kvality na výrobných linkách,
- prediktívnou údržbou strojov na základe snímačov vibrácií a teploty,
- inteligentným riadením energie a klímy v budovách,
- klasifikáciou medicínskych signálov a autonómnymi mobilnými robotmi.

Tradičná výučba strojového učenia však často padá do jedného z dvoch extrémov:
1. **Prílišná matematická abstrakcia:** viacrozmerná analýza, parciálne derivácie a maticová algebra, ktoré žiakov odborných škôl odradia.
2. **Knižnice ako „čierna skrinka“:** volanie hotových funkcií v Pythone (`model.fit()`, `model.predict()`), pri ktorom žiaci vidia výsledok, ale nechápu, čo algoritmus naozaj robí.

### 1.2. Pedagogický prístup: Od vizuálnej intuície k inžinierskemu porozumeniu
Balík **AI for VET** prináša **interaktívne bádateľské učenie (Active Inquiry-Based Learning)**:
- **Experimentovanie v reálnom čase:** Žiak mení hyperparametre posuvníkmi a tlačidlami a hneď vidí, ako algoritmus reaguje.
- **Beh v prehliadači bez inštalácie:** Netreba Python, ovládače GPU ani servery – simulácie bežia celé v prehliadači, aj na slabších školských počítačoch, tabletoch a mobiloch.
- **Dvojfázový pedagogický cyklus:**
  1. *Fáza skúmania (lekcia):* Žiak voľne mení parametre a hľadá zákonitosti.
  2. *Fáza overenia (test):* Žiak rieši praktický problém, v ktorom musí algoritmus dostať do požadovanej tolerancie, a Moodle riešenie automaticky ohodnotí.

### 1.3. Architektúra balíka pre Moodle
Balík tvoria tri spolupracujúce pluginy:
- **`moodle-filter_aisim`** (textový filter a simulácie): zobrazí 10 interaktívnych simulácií všade, kde sa v obsahu objaví značka `data-aisim`.
- **`moodle-qtype_aisim`** (typ otázky a automatické hodnotenie): vkladá interaktívne úlohy do testov Moodle a po odovzdaní vyhodnotí dosiahnuté hodnoty.
- **`moodle-tiny_aisim`** (plugin editora TinyMCE): dialóg, v ktorom učiteľ vloží, nastaví a dvojklikom upraví simuláciu bez písania HTML.

Simulácie sa zobrazujú v jazyku používateľa v Moodle: obsahujú angličtinu, srbčinu (latinku aj cyriliku), slovenčinu a portugalčinu.

---

## 2. Pedagogický sprievodca 13 simuláciami

Simulácie idú od jednoduchších k náročnejším. Prvých päť pokrýva základy (optimalizácia, hodnotenie modelu, preučenie, klasifikácia, zhlukovanie), moduly 6–9 na ne nadväzujú neurónovými sieťami, spracovaním obrazu, učením z odmien a zodpovednou prácou s dátami. Modul 10, logistická regresia, je mostom medzi regresiou a neurónovými sieťami: dá sa zaradiť hneď po matici zámen alebo pred perceptrón. Modul 11, krížová validácia, patrí k polynomiálnej regresii: ukazuje, ako sa hyperparameter volí poctivo. Modul 12, príprava dát, patrí úplne na začiatok: pred akýmkoľvek modelom sa pozrite na dáta. Modul 13, rekurentná neurónová sieť, nasleduje po neurónovej sieti: pridáva pamäť pre postupnosti.

### 2.1. Modul 1: Gradientný zostup a optimalizácia modelu (`gradient_descent`)
- **Kľúčové pojmy:** nákladová funkcia J(w), rýchlosť učenia α, konvexné a nekonvexné funkcie, lokálne minimá.
- **Pedagogická metafora:** Turista zostupuje z hory v hustej hmle. Nevidí mapu, len sklon terénu pod nohami. V každom kroku nahmatá smer najväčšieho klesania a urobí krok tým smerom. Dĺžka kroku je **rýchlosť učenia (α)**.
- **Praktické využitie:** ladenie regulátorov v mechatronike; učenie neurónových sietí (backpropagation).
- **Čo žiaci pozorujú:**
  1. *Malá rýchlosť učenia (α = 0,01):* algoritmus postupuje veľmi pomaly.
  2. *Vhodná rýchlosť učenia (α = 0,10 – 0,20):* rýchla a stabilná konvergencia do minima.
  3. *Príliš veľká rýchlosť učenia (α ≥ 1,05):* oscilácie a divergencia – algoritmus preskakuje dno a „uletí“.
  4. *Nekonvexná funkcia:* počiatočný bod (w₀) rozhoduje, v ktorom lokálnom minime model skončí.
- **Typická úloha do testu:**
  > *„Minimum konvexnej nákladovej funkcie je J(w) = 1,00. Nastavte rýchlosť učenia (α) a spustite algoritmus tak, aby náklady klesli na J(w) ≤ 1,01.“*

---

### 2.2. Modul 2: Matica zámen a rozhodovací prah (`confusion_matrix`)
- **Kľúčové pojmy:** hodnotenie binárneho klasifikátora, nevyvážené triedy, prah θ, TP, FP, FN, TN, presnosť (Precision), citlivosť (Recall), F1-skóre.
- **Prečo správnosť (Accuracy) klame:** Ak je chybných len 2 % výrobkov, model, ktorý všetko označí za dobré, má **správnosť 98 %**, a pritom nezachytí ani jeden chybný kus.
- **Význam metrík:**
  - *Prah (θ):* pravdepodobnosť, nad ktorou kus označíme za chybný.
  - *Presnosť (Precision):* koľko z kusov označených za chybné je naozaj chybných (málo planých poplachov).
  - *Citlivosť (Recall):* koľko zo všetkých chybných kusov model zachytil (žiadny chybný kus nejde k zákazníkovi).
  - *F1-skóre:* harmonický priemer presnosti a citlivosti.
- **Čo žiaci pozorujú:** Posun prahu θ doľava zvyšuje citlivosť na úkor presnosti; posun doprava zvyšuje presnosť, ale model prepúšťa chyby.
- **Typická úloha do testu:**
  > *„Nastavte prah θ tak, aby klasifikátor zachytil väčšinu chybných kusov bez priveľa planých poplachov: citlivosť ≥ 0,80 a presnosť ≥ 0,70.“*

---

### 2.3. Modul 3: Polynomiálna regresia a regularizácia (`regression`)
- **Kľúčové pojmy:** nelineárne závislosti, stupeň polynómu d, nedoučenie (*underfitting*), preučenie (*overfitting*), chyba na trénovacích a testovacích dátach, L₂ regularizácia (λ).
- **Pedagogická metafora:** Nedoučenie je, keď žiak prečíta len nadpisy kapitol. Preučenie je, keď sa naspamäť naučí každé písmeno zbierky vrátane tlačových chýb – a pri zmene jedného čísla v úlohe nevie nič.
- **Praktické využitie:** kalibrácia snímačov teploty a tlaku; predpoveď zaťaženia transformátorov podľa času a teploty.
- **Čo žiaci pozorujú:**
  - d = 1: priamka nedokáže sledovať vlnu (vysoká chyba na tréningu aj teste).
  - d = 3: hladká krivka dobre vystihuje skutočný signál.
  - d = 5 a λ = 0: krivka prechádza každým bodom so šumom a na okrajoch divoko kmitá.
  - λ = 0,01 – 0,10: penalizácia veľkých váh krivku vyhladí a obnoví zovšeobecnenie.
- **Typická úloha do testu:**
  > *„Pre periodický signál so šumom zvoľte stupeň polynómu najviac 7 tak, aby model vystihol trend bez preučenia, s validačnou chybou MSE ≤ 0,19.“*

---

### 2.4. Modul 4: Nelineárne rozhodovacie hranice – k-NN a rozhodovací strom (`knn_tree`)
- **Kľúčové pojmy:** učenie s učiteľom, k najbližších susedov, rozhodovací strom, rozhodovacia hranica, hyperparametre k a maximálna hĺbka stromu.
- **Porovnanie:**
  - *k-NN:* „Povedz mi, kto sú tvoji priatelia, a poviem ti, kto si.“ Nový bod dostane triedu, ktorá prevláda medzi k najbližšími susedmi – hranica je oblá.
  - *Rozhodovací strom:* séria otázok „ak – tak – inak“, ktoré rozdelia rovinu na obdĺžniky rovnobežné s osami.
- **Vplyv hyperparametrov:** k = 1 je citlivé na šum (ostrovčeky okolo jednotlivých bodov); k = 15 hranicu príliš zjednoduší. Plytký strom nestačí, hlboký strom rozkúskuje rovinu na drobné obdĺžniky okolo každého bodu.

---

### 2.5. Modul 5: Učenie bez učiteľa a zhlukovanie K-Means (`kmeans`)
- **Kľúčové pojmy:** zoskupovanie bez označení, centroidy, euklidovská vzdialenosť, inercia (súčet štvorcov vzdialeností), postupná konvergencia.
- **Praktické využitie:** rozdelenie odberateľov elektriny podľa profilu spotreby; kompresia farieb a analýza satelitných snímok.
- **Čo žiaci pozorujú:** Tlačidlom „Ďalší krok“ sledujú dva striedajúce sa kroky:
  1. *Priradenie:* každý bod dostane farbu najbližšieho centroidu.
  2. *Aktualizácia:* každý centroid sa presunie do ťažiska svojich bodov.
- **Vplyv K:** K = 2 na dátach s tromi zhlukmi ukáže, ako jeden centroid „pokrýva“ dve skupiny; inercia prestane výrazne klesať pri skutočnom počte zhlukov.

---

### 2.6. Modul 6: Perceptrón a neurónová sieť (`neural_network`)
- **Kľúčové pojmy:** perceptrón, lineárna oddeliteľnosť, skrytá vrstva, aktivačné funkcie (sigmoid, tanh, ReLU), epochy, chybová funkcia (log-loss), rýchlosť učenia, počiatočné váhy.
- **Pedagogická metafora:** Jeden neurón je pravítko: nakreslí len jednu priamku a povie „na tejto strane červené, na tej modré“. Skrytá vrstva je tím pravítok, ktorých priamky výstupný neurón spojí do ohnutej hranice – ako keď z papiera vystrihnete kruh niekoľkými rovnými strihmi. Viac neurónov, jemnejšie strihy.
- **Praktické využitie:** rozlíšenie dobrých a chybných dielov z niekoľkých meraní snímačov. Neurónové siete sú základom rozpoznávania reči a obrazu aj veľkých jazykových modelov – tu je rovnaká myšlienka v najmenšej podobe.
- **Čo žiaci pozorujú:**
  1. *Lineárne oddeliteľné dáta, 0 skrytých vrstiev:* perceptrón rýchlo nájde priamku a dosiahne takmer 100 %.
  2. *XOR (prekrížené kvadranty), 0 skrytých vrstiev:* správnosť zostáva okolo 50 %, nech trénuje akokoľvek dlho – žiadna priamka triedy neoddelí. Simulácia to povie aj v rade.
  3. *XOR s jednou skrytou vrstvou s 2–4 neurónmi:* hranica sa ohne a správnosť presiahne 98 %. Hrúbka čiar na nákrese siete ukazuje, ktoré spojenia zosilneli.
  4. *Kruh v kruhu:* s 2 neurónmi sieť často uviazne; s 3–4 neurónmi a tlačidlom „Nové váhy“ nájde uzavretú hranicu okolo vnútornej triedy.
  5. *Príliš veľká rýchlosť učenia (napr. 3):* chyba „vybuchne“ – rovnaké poučenie ako pri gradientnom zostupe, lebo trénovanie siete *je* gradientný zostup.
- **Typická úloha do testu:**
  > *„Na dátach XOR zostáva perceptrón na 50 %. Pridajte skrytú vrstvu a dosiahnite správnosť ≥ 98 % s najviac 4 skrytými neurónmi.“*

---

### 2.7. Modul 7: Konvolúcia a obrazové filtre (`convolution`)
- **Kľúčové pojmy:** obrázok ako tabuľka čísel, jadro 3×3, deliteľ, rozmazanie, zaostrenie, detekcia hrán (Sobel, Laplace), absolútna hodnota, konvolučné neurónové siete (CNN).
- **Pedagogická metafora:** Jadro je malá „lupa s pravidlom“, ktorá sa posúva po obrázku. Na každom mieste pozrie 9 pixelov, vynásobí ich svojimi váhami a zapíše jedno číslo. Ak sú všetky váhy rovnaké, lupa priemeruje (rozmazáva). Ak jednu stranu odčíta od druhej, „vidí“ len miesta, kde sa jas prudko mení – hrany.
- **Praktické využitie:** optická kontrola kvality (škrabance, praskliny, hrany dielov na linke); odstránenie šumu zo snímok kamier a medicínskych snímok. Prvé vrstvy každej siete na rozpoznávanie obrazu (CNN) sa učia práve takéto jadrá.
- **Čo žiaci pozorujú:**
  1. *Ťuknutie na pixel* ukáže celý výpočet: 9 hodnôt, 9 váh, súčet a delenie. Toto je kľúčový moment hodiny – konvolúcia prestáva byť „mágia“.
  2. *Rozmazanie (priemer) a Gaussovo rozmazanie:* šum zmizne, ale zmäknú aj hrany.
  3. *Súčet váh / deliteľ ≠ 1:* celý obrázok zosvetlie alebo stmavne – simulácia na to upozorní.
  4. *Sobel bez absolútnej hodnoty:* hrany zo svetlého do tmavého dávajú záporné hodnoty a zobrazia sa ako čierne. S |x| sú viditeľné oba druhy hrán.
  5. *Cieľový obrázok a zhoda (R²):* žiaci hádajú alebo sami skladajú filter, ktorý dá zadaný výsledok.
- **Typická úloha do testu:**
  > *„Obrázok je plný šumu a hotové filtre sú skryté. Zadajte váhy jadra a deliteľ tak, aby výstup vyzeral ako cieľ, so zhodou ≥ 95 %.“* (Jedno riešenie: váhy 1 2 1 / 2 4 2 / 1 2 1 a deliteľ 16.)

---

### 2.8. Modul 8: Posilňované učenie – Q-learning (`q_learning`)
- **Kľúčové pojmy:** agent, stav, akcia, odmena, Q-tabuľka, epizóda, rýchlosť učenia α, diskontný faktor γ, skúmanie ε (skúmanie verzus využívanie), politika.
- **Pedagogická metafora:** Výcvik psa: nikto mu nevysvetľuje pravidlá, len dostane maškrtu, keď urobí správnu vec. Po mnohých pokusoch si pamätá, ktorý ťah sa v ktorej situácii oplatí. Agent v simulácii robí to isté: −0,1 za každý krok, +10 v cieli, −10 pri páde z útesu.
- **Praktické využitie:** riadenie robotov a autonómnych vozidiel; riadenie energie (nabíjanie batérií, kúrenie) a semaforov; AI v hrách od šachu a go po videohry.
- **Čo žiaci pozorujú:**
  1. *Bludisko:* agent najprv blúdi náhodne. Po niekoľkých desiatkach epizód farby polí a šípky vytvoria „rázcestník“ k cieľu a naučená cesta sa skráti na 10 krokov.
  2. *Útes s veľkým ε (napr. 0,4):* agent sa naučí najkratšiu cestu tesne popri okraji, ale počas tréningu náhodne padá – priemerná odmena je nízka. S ε = 0,05 padá oveľa menej.
  3. *Blízko alebo ďaleko s γ = 0,5:* agent si berie malý poklad T (+3), lebo budúcnosť má pre neho malú hodnotu. Vyššie γ (napr. 0,9) zvýši hodnotu veľkého cieľa G (+10) – ale len ak ho agent vôbec objavil, na čo treba dostatočné ε.
  4. *Ťuknutie na pole* ukáže Q-hodnoty všetkých štyroch ťahov – žiaci vidia čísla za šípkou.
- **Typická úloha do testu:**
  > *„Agent si vyberá malý poklad T (+3) neďaleko. Zmeňte diskontný faktor γ a skúmanie ε tak, aby sa naučil ísť k veľkému cieľu G (+10) za 4 kroky, najviac za 1000 epizód. Vysvetlite úlohu každého parametra.“*

---

### 2.9. Modul 9: Rozdelenie dát a skreslenie (`data_bias`)
- **Kľúčové pojmy:** trénovacia a testovacia množina, zovšeobecnenie, zapamätanie (k = 1), reprezentatívna vzorka, skreslenie dát (*data bias*), správnosť po skupinách, spravodlivosť modelu.
- **Pedagogická metafora:** Písomka zložená len z príkladov, ktoré žiak už videl, nehovorí nič. A ak cvičenia aj písomka pochádzajú z učebnice bez jediného príkladu z geometrie, žiak dostane jednotku – a zlyhá pri prvom trojuholníku. Testovacia množina odhalí len chyby, ktoré jej vzorka obsahuje.
- **Praktické využitie:** model kontroly kvality natrénovaný na dieloch z jednej továrne a použitý v inej (iný stroj, iný materiál). Systémy na schvaľovanie úverov, nábor či medicínsku diagnostiku, ktoré horšie fungujú pre skupiny zriedkavé v dátach. Európsky AI Act vyžaduje práve kvalitné, reprezentatívne dáta a kontrolu pre rôzne skupiny.
- **Čo žiaci pozorujú:**
  1. *Podiel továrne B 5 %:* test ukazuje okolo 95 %, ale skutočná správnosť pre továreň B je oveľa nižšia. Priemer skryje zriedkavú skupinu.
  2. *Vyšší podiel B (napr. 30–50 %):* správnosť pre B prudko stúpne, hoci celkový počet vzoriek je rovnaký – dôležitejšie je, *aké* dáta, než *koľko* dát.
  3. *„B sa správa podobne“:* malé zastúpenie neškodí, keď sa skupiny nelíšia – problém vzniká, keď sa líšia a nie sú zastúpené.
  4. *k = 1:* správnosť na tréningu je 100 % (model si zapamätal každý bod), čo o jeho kvalite nehovorí nič.
  5. *Malá testovacia množina:* výsledok testu skáče s každou novou vzorkou (🎲) – malý test je nespoľahlivé meradlo.
- **Typická úloha do testu:**
  > *„Zber dát je drahý. S najviac 120 vzorkami dosiahnite skutočnú správnosť oboch tovární ≥ 88 %. Čo je dôležitejšie: koľko dát, alebo aké dáta?“*

---

### 2.10. Modul 10: Logistická regresia (`logistic`)
- **Kľúčové pojmy:** Binárna klasifikácia, vážený súčet z = w₁·x₁ + w₂·x₂ + b, sigmoida σ(z), pravdepodobnosť, rozhodovacia hranica, rozhodovací prah θ, log-strata, gradientný zostup, presnosť a záchyt, hranice lineárneho modelu.
- **Pedagogická metafora:** Majster rozhoduje, či sa stroj pokazí, podľa dvoch ukazovateľov: vibrácií a teploty. Každý ukazovateľ má svoju dôležitosť (váhy) a majster má aj vlastnú mieru opatrnosti (posun, *bias*). Sigmoida premení „súčet obáv“ na pravdepodobnosť medzi 0 a 1 a prah θ je chvíľa, keď majster zastaví linku.
- **Praktické využitie:** Prediktívna údržba (porucha podľa hodnôt senzorov), predpoveď výsledku skúšky, hodnotenie úverov, medicínsky skríning. Logistická regresia je v priemysle často prvým modelom, lebo každú váhu možno prečítať a vysvetliť.
- **Čo žiaci v simulácii pozorujú:**
  1. *Ručné posúvanie w₁, w₂ a b:* Váhy otáčajú hranicu, posun ju posúva. Všetky body, kde σ(z) = θ, ležia na jednej priamke.
  2. *Trénovať:* Gradientný zostup sám otočí priamku k dátam; krivka straty rýchlo klesne a potom sa vyrovná.
  3. *Ťuknutie na bod:* Výpočet ukáže z, σ(z), porovnanie s θ a predpoveď – model je len vzorec.
  4. *Posúvanie θ:* Nízky prah zachytí každú poruchu (záchyt 100 %), ale spôsobí falošné poplachy; vysoký prah dá istejšie, ale zriedkavejšie poplachy. Váhy sa nemenia – mení sa len rozhodnutie.
  5. *Dáta o skúške:* Triedy sa prekrývajú, takže žiadna priamka nie je dokonalá (asi 80–82 %); lepší model potrebuje lepšie dáta, nie dlhší tréning.
  6. *Prstenec:* Priamka nedokáže oddeliť prstenec (najviac asi 68 %). To je motivácia pre nové príznaky (x₁² + x₂²) a pre skrytú vrstvu neurónovej siete (Modul 6).
- **Typická úloha v teste:**
  > *„Prehliadnutá porucha zastaví celú výrobnú linku. Model je už natrénovaný: posúvajte len prah θ tak, aby zachytil každú poruchu (záchyt 100 %) s presnosťou aspoň 85 %.“*

---

### 2.11. Modul 11: Krížová validácia a hyperparametre (`cross_validation`)
- **Kľúčové pojmy:** Hyperparameter, chyba na tréningu a validačná chyba, k-násobná krížová validácia, priemer a rozptyl chýb častí, nedoučenie a preučenie, testovacia množina použitá len raz.
- **Pedagogická metafora:** Tréner vyberá zostavu na finále. Ak hráčov hodnotí len podľa tréningu (kde poznajú každé cvičenie), vždy vyberie toho, kto sa cvičenia naučil naspamäť. Preto odohrá niekoľko priateľských zápasov, každý s iným súperom (časti), a vyberie tím, ktorý je v priemere najlepší. Finále (testovacia množina) sa hrá len raz – potom sa výber už nemení.
- **Praktické využitie:** Voľba zložitosti akéhokoľvek modelu: stupeň polynómu, k pri k-NN, hĺbka stromu, počet neurónov, sila regularizácie. V priemysle sa model vyberá presne takto a samostatná testovacia množina sa necháva na záverečnú správu.
- **Čo žiaci v simulácii pozorujú:**
  1. *Posúvanie stupňa:* Chyba na tréningu (modrá) vždy klesá, chyba krížovej validácie (červená) klesá a potom znova rastie – U-krivka. Najlepší stupeň je označený ★.
  2. *Stĺpce častí:* Každá časť dáva inú chybu; ich priemer je odhad a rozptyl ukazuje, nakoľko mu veriť.
  3. *k = 2 a 🎲 Nové rozdelenie:* Pri málo častiach sa každý model učí z polovice dát a najlepší stupeň skáče od rozdelenia k rozdeleniu, najmä pri zašumených dátach. Pri k = 10 sa ustáli.
  4. *🔒 Záverečný test:* Chyba na teste pre zvolený stupeň je blízko odhadu krížovej validácie, ale nie je rovnaká – ide o novú vzorku dát.
  5. *Ladenie po teste:* Keď sa výsledok testu raz uvidí, nová voľba podľa neho mení test na validačnú množinu; simulácia to pripomína.
- **Typická úloha v teste:**
  > *„S aspoň 5 časťami zvoľte stupeň polynómu tak, aby CV MSE bola najviac 0,11. Prečo chyba na tréningu nemôže zvoliť stupeň?“*

---

### 2.12. Modul 12: Príprava dát a EDA (`data_prep`)
- **Kľúčové pojmy:** Prieskumná analýza dát (minimum, maximum, priemer, medián, histogram), chýbajúce hodnoty (vyhodenie, nahradenie nulou, priemerom alebo mediánom), odľahlé hodnoty spôsobené chybou, škálovanie (min-max, z-score), postup prípravy naučený na trénovacích dátach a použitý na nové dáta.
- **Pedagogická metafora:** Kuchár kontroluje suroviny pred varením: pokazené vajce (nemožné meranie), chýbajúca surovina (chýbajúca hodnota) a recept, ktorý mieša gramy a kilogramy (rôzne mierky), pokazia jedlo, nech je recept (model) akokoľvek dobrý.
- **Praktické využitie:** Dáta zo senzorov s výpadkami a chybami v jednotkách, záznamy pacientov s chýbajúcimi vyšetreniami, každá tabuľka, ktorá mieša veľké a malé čísla. V praxi väčšina času v projekte strojového učenia ide práve na tento krok.
- **Čo žiaci v simulácii pozorujú:**
  1. *Tabuľka a histogramy:* Maximálne otáčky 45 000 ot/min sú nemožné – chyba v jednotkách. Pri „Nahradiť nulou“ sa minimum vibrácií zmení na 0 a priemer klesne: vymyslené merania vidno v štatistikách.
  2. *Bez škálovania:* V zobrazení „Čo vidí model k-NN“ (rovnaká mierka na oboch osiach) ležia všetky body na plochej čiare – rozhoduje sama premenná s veľkými číslami a správnosť je asi 50 %.
  3. *Min-max s odľahlými hodnotami:* Niekoľko extrémnych hodnôt roztiahne mierku a stlačí bežné hodnoty do úzkeho pásu; pomôže odstránenie odľahlých hodnôt (alebo z-score).
  4. *Chýbajúce hodnoty:* Vyhodenie riadkov stráca dáta, viac z triedy, ktorej senzor zlyháva častejšie; nahradenie mediánom zachová všetky riadky bez vymýšľania extrémnych hodnôt.
  5. *Jeden postup pre všetky dáta:* Minimum, maximum, priemer či medián sa naučia na trénovacích dátach a nezmenené sa použijú na každý nový príklad.
- **Typická úloha v teste:**
  > *„Model háda takmer náhodne. Pozrite si štatistiky a to, čo vidí k-NN, a pripravte dáta tak, aby správnosť na testovacích dátach dosiahla aspoň 85 %.“*

---

### 2.13. Modul 13: Rekurentná neurónová sieť (`rnn`)
- **Kľúčové pojmy:** Postupnosti, skrytý stav, ktorý nesie pamäť z kroku do kroku (h_t = tanh(w·h_(t−1) + u·x_t), y_t = v·h_t), rovnaké váhy v každom kroku (sieť rozvinutá v čase), zabúdanie, spätné šírenie v čase, miznúci gradient a prečo majú siete LSTM a GRU brány.
- **Pedagogická metafora:** Šepot podávaný pozdĺž radu žiakov: každý žiak zopakuje, čo počul, o niečo tichšie (w < 1). Po niekoľkých žiakoch sa prvé slová stratia – iba ak ich všetci opakujú presne rovnako nahlas (w = 1), zostanú.
- **Praktické využitie:** Predpoveď teploty, spotreby energie alebo zaťaženia stroja z posledných hodín, čítanie signálov senzorov, text a reč. Priemerovanie zašumeného senzora je najjednoduchšie použitie pamäte.
- **Čo žiaci v simulácii pozorujú:**
  1. *Teplomer bez pamäte:* Pri w ≈ 0 sieť iba škáluje posledné zašumené meranie (chyba asi 0,072); pri w okolo 0,35 priemeruje niekoľko hodín a chyba klesne asi na 0,059. Tlačidlo Trénovať to nájde po niekoľkých kliknutiach.
  2. *Stĺpce vplyvu:* Každý stĺpec je vplyv jedného predchádzajúceho vstupu na posledný výstup. V každom kroku sa násobí w, takže pri w = 0,5 rozhoduje iba niekoľko posledných vstupov.
  3. *Zapamätaj si prvú hodnotu:* Úloha s pamäťou potrebuje w blízko 1; malé u drží tanh v jeho rovnej časti a veľké v hodnotu vráti (L = 20: w = 1, u = 0,3, v = 4 dajú chybu asi 0,003).
  4. *w > 1:* Stav narastie na ±1 a tam uviazne: sieť si ponechá iba znamienko, nie veľkosť hodnoty.
  5. *Miznúci gradient:* Pri L = 20 a w = 0,5 je veľkosť gradientu ≈ 0 a Trénovať nič nezmení – chyba sa nedostane k prvému kroku. To je problém, ktorý riešia LSTM a GRU.
- **Typická úloha v teste:**
  > *„Postupnosť má 20 krokov. Stlačte Trénovať a sledujte gradient: prečo nepomáha? Potom ručne nastavte sieť na chybu najviac 0,01 s aspoň 20 zapamätanými vstupmi.“*

---

## 3. Technický návod pre učiteľov

### 3.1. Vkladanie simulácií vizuálnym editorom TinyMCE (`tiny_aisim`)
1. V paneli nástrojov editora TinyMCE kliknite na **ikonu AI simulácie** (alebo v ponuke zvoľte *Vložiť > AI Simulation...*).
2. Nastavte parametre:
   - Zvoľte jednu z 13 simulácií.
   - Nastavte scenár, prah, rýchlosť učenia, stupeň, počet zhlukov, architektúru siete, obrázok a filter, mapu agenta alebo podiel skupiny v dátach.
   - Zaškrtnite **Len na čítanie**, ak chcete, aby žiaci simuláciu len pozorovali.
   - V spodnej časti dialógu vidíte živý náhľad.
3. Kliknite na **Vložiť simuláciu**.
4. **Úprava existujúcej simulácie:** stačí **dvojklik** na kartu simulácie v editore – dialóg sa otvorí s jej nastaveniami.

---

### 3.2. Tvorba automaticky hodnotených úloh (`qtype_aisim`)
1. V teste Moodle zvoľte **Otázky > Pridať novú otázku**.
2. Vyberte **AI Simulation (`aisim`)**.
3. Zadajte názov otázky, text úlohy, body a penalizáciu.
4. V sekcii **Nastavenia AI simulácie**:
   - Zvoľte typ simulácie a scenár.
   - Nastavte cieľovú metriku 1 a 2 (napr. citlivosť `0.80` a presnosť `0.70` pre maticu zámen). Ich význam závisí od simulácie – pozrite pomocníka (?) pri poliach a tabuľku v prílohe.
   - Tolerancia sa používa pri gradientnom zostupe (polovica bodov do cieľa + tolerancie) a pri konvolúcii (polovica bodov do cieľa − tolerancie).
   - Zadajte spätnú väzbu pre správnu, čiastočne správnu a nesprávnu odpoveď.
5. Uložte otázku. Počas testu Moodle zobrazí interaktívnu simuláciu a po odovzdaní automaticky vypočíta body.

---

### 3.3. Import hotovej banky otázok (Moodle XML)
Repozitár `moodle-qtype_aisim` obsahuje 39 hotových úloh – po tri pre každú z 13 simulácií (základná, stredná a pokročilá úroveň):
1. Stiahnite [`questions_aiforvet.xml`](https://github.com/blagojevicboban/moodle-qtype_aisim/raw/master/demo/questions_aiforvet.xml).
2. V kurze otvorte **Banka otázok > Import**.
3. Zvoľte **formát Moodle XML**.
4. Nahrajte `questions_aiforvet.xml` a kliknite na **Importovať**.
5. Vytvorí sa kategória *„AI za VET - Baza simulacionih zadataka (39 pitanja)“* so všetkými otázkami pripravenými do testu. (Texty otázok sú v srbčine; upravte ich podľa potreby.)

---

### 3.4. AISim Lab: simulácie bez Moodle
Na [sim.aiforvet.eu](https://sim.aiforvet.eu/) beží všetkých 13 simulácií samostatne – bez účtu a bez zberu údajov – po slovensky, anglicky, srbsky a portugalsky. Každá simulácia má krátke vysvetlenie a tri výzvy.
- **Domáca úloha a mobily:** žiaci môžu cvičiť mimo Moodle; postup vo výzvach sa ukladá len v ich prehliadači.
- **„Kopírovať odkaz na toto nastavenie“:** vytvorí odkaz presne na tento stav simulácie – vhodné pre projektor alebo správu žiakom.
- **„Vložiť do Moodle“:** dá hotový HTML kód tohto nastavenia na vloženie do ľubovoľného textu v Moodle (potrebný je `filter_aisim`).

---

## 4. Metodické odporúčania a organizácia vyučovania

### 4.1. Štruktúra 90-minútovej praktickej hodiny
```
[00-15 min] Skutočný problém z praxe a priemyselný kontext
     │
     ▼
[15-35 min] Riadené skúmanie simulácie v lekcii
     │
     ▼
[35-50 min] Diskusia vo dvojiciach, hypotézy a pojmy
     │
     ▼
[50-75 min] Samostatné riešenie úloh v automaticky hodnotenom teste
     │
     ▼
[75-90 min] Reflexia a prepojenie s odborom
```

### 4.2. Formatívne hodnotenie a učenie z chýb
- V teste nastavte správanie otázok na **Okamžitá spätná väzba** alebo **Interaktívny režim s viacerými pokusmi**.
- Žiak nemá byť trestaný za skúšanie hraničných hodnôt, ale odmenený za pochopenie, prečo algoritmus diverguje alebo chybne klasifikuje.
- Pokročilé úlohy z banky (napr. Q-learning a skreslenie dát) končia otázkou „Vysvetlite…“ – využite ju na krátku ústnu alebo písomnú odpoveď po teste.

### 4.3. Diferencované vyučovanie
- **Základná úroveň:** posuvníky a sledovanie, či odznak cieľa zozelenie. Pri konvolúcii žiaci vyberajú hotové filtre, pri Q-learningu sledujú, ako sa agent učí bludisko.
- **Pokročilá úroveň:** kompromis presnosť – citlivosť, porovnanie regularizovaných modelov, ručné skladanie jadier, hľadanie najmenšej siete, ktorá úlohu vyrieši, a vysvetlenie vplyvu γ a ε na správanie agenta.

### 4.4. Odporúčané poradie simulácií
| Hodina / téma | Simulácie | Kľúčová myšlienka |
| :--- | :--- | :--- |
| 1. Ako sa model učí | Gradientný zostup | Učenie = zmenšovanie chyby po malých krokoch |
| 1a. Najprv dáta | Príprava dát (EDA) | Pozrite sa na dáta; chýbajúce hodnoty, odľahlé hodnoty a škálovanie rozhodnú skôr než model |
| 1b. Od skóre k rozhodnutiu | Logistická regresia | Pravdepodobnosť, priama hranica a prah zvolený podľa ceny každej chyby |
| 2. Ako model meriame | Matica zámen, Rozdelenie dát a skreslenie | Správnosť nestačí; test musí byť férový a reprezentatívny |
| 3. Zložitosť modelu | Polynomiálna regresia, k-NN a rozhodovací strom | Príliš jednoduchý verzus preučený model |
| 3b. Poctivá voľba modelu | Krížová validácia | Hyperparametre sa volia na validačných častiach; testovacia množina sa použije raz |
| 4. Učenie bez odpovedí | K-Means | Zoskupovanie bez označení |
| 5. Neurónové siete a obraz | Perceptrón a neurónová sieť, Konvolúcia | Od jedného neurónu k rozpoznávaniu obrazu |
| 5b. Pamäť pre postupnosti | Rekurentná neurónová sieť | Stav prenášaný z kroku do kroku; zabúdanie a miznúci gradient vedú k LSTM a GRU |
| 6. Učenie z odmien | Q-learning | Skúmanie verzus využívanie, hodnota budúcnosti |

---

## 5. Príloha a prehľad vzorcov

| Simulácia | Kľúčové parametre | Model / vzorec | Typické kritérium úspechu |
| :--- | :--- | :--- | :--- |
| **Gradientný zostup** | α (krok), w₀ (začiatok) | w ← w − α · J′(w) | J(w) ≤ 1,01 (minimum je 1,00) |
| **Matica zámen** | θ (prah) | F1 = 2 · P · R / (P + R) | Citlivosť ≥ 0,80 a presnosť ≥ 0,70 |
| **Polynomiálna regresia** | d (stupeň), λ (regularizácia) | J(w) = (1/2m) · Σ(h(x) − y)² + (λ/2) · Σw² | Validačná MSE ≤ 0,19 pri d ≤ 7 |
| **k-NN** | k (počet susedov) | ŷ = najčastejšia trieda medzi k najbližšími susedmi | Správnosť ≥ 98 % |
| **K-Means** | K (počet zhlukov) | min Σ ‖x − μ‖² (súčet štvorcov vzdialeností k centroidom) | Konvergencia pri K = 3 a WCSS ≤ 10 |
| **Neurónová sieť** | vrstvy, neuróny, aktivácia, rýchlosť učenia | a = f(w · x + b); w ← w − α · ∂L/∂w | Správnosť ≥ 98 % s najviac 4 skrytými neurónmi |
| **Konvolúcia** | jadro 3×3, deliteľ, \|x\| | y = Σ (váha · pixel) / deliteľ | Zhoda s cieľom (R²) ≥ 95 % |
| **Q-learning** | α, γ, ε, počet epizód | Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s′,·) − Q(s,a)) | Najkratšia cesta (napr. 10 krokov) za ≤ 200 epizód |
| **Rozdelenie dát a skreslenie** | veľkosť vzorky, podiel skupiny B, test %, k | správnosť najslabšej skupiny = min(správnosť A, správnosť B) | Obe továrne ≥ 88 % s ≤ 120 vzorkami |
| **Logistická regresia** | w₁, w₂, b, θ, α | σ(z) = 1 / (1 + e^(−z)), z = w₁·x₁ + w₂·x₂ + b; trieda 1, ak σ(z) ≥ θ | Správnosť ≥ 95 % (machine); záchyt 100 % pri presnosti ≥ 85 % |
| **Krížová validácia** | d (stupeň), k (časti) | CV chyba = (1/k) · Σ MSE časti i, každá časť odložená raz | CV MSE ≤ 0,11 pri k ≥ 5 (sine) |
| **Príprava dát** | chýbajúce hodnoty, odľahlé hodnoty, škálovanie, k | min-max: x′ = (x − min) / (max − min); z-score: x′ = (x − priemer) / σ | Správnosť ≥ 85 % (machine); správnosť ≥ 90 % a záchyt ≥ 85 % (patients) |
| **Rekurentná neurónová sieť** | w (pamäť), u, v, dĺžka postupnosti L | h_t = tanh(w·h_(t−1) + u·x_t), y_t = v·h_t | MSE ≤ 0,065 (temperature); MSE ≤ 0,01 s pamäťou ≥ 20 vstupov (memory, L = 20) |

#### Význam polí „Cieľová metrika“ v otázkach (`qtype_aisim`)
| Simulácia | Cieľová metrika 1 | Cieľová metrika 2 | Polovica bodov |
| :--- | :--- | :--- | :--- |
| Gradientný zostup | najvyššie povolené náklady J(w) | – | náklady do cieľa + tolerancie |
| Matica zámen | minimálna citlivosť (Recall) | minimálna presnosť (Precision) | splnená jedna podmienka |
| Polynomiálna regresia | najvyššia validačná MSE | najvyšší stupeň d | splnená jedna podmienka |
| k-NN a rozhodovací strom | minimálna správnosť na teste | – | správnosť do 5 % pod cieľom |
| K-Means | najvyššia inercia (WCSS) | presný počet zhlukov K (0 = ľubovoľný) | splnená jedna podmienka |
| Neurónová sieť | minimálna správnosť | najviac skrytých neurónov (0 = bez obmedzenia) | splnená jedna podmienka |
| Konvolúcia | minimálna zhoda (R²) | – | zhoda do cieľa − tolerancie |
| Q-learning | najviac krokov do cieľa | najviac epizód tréningu (0 = bez obmedzenia) | agent dôjde do cieľa, ale dlhšou cestou alebo po viacerých epizódach |
| Rozdelenie dát a skreslenie | minimálna správnosť najslabšej továrne | najväčší počet vzoriek (0 = bez obmedzenia) | správnosť splnená, ale s priveľa vzorkami |
| Logistická regresia | minimálna správnosť | minimálny záchyt (0 = bez obmedzenia); pole Tolerancia určuje minimálnu presnosť | správnosť splnená, ale záchyt alebo presnosť nie |
| Krížová validácia | najvyššia CV MSE zvoleného stupňa | najmenší počet častí k (0 = bez obmedzenia) | MSE splnená, ale s príliš málo časťami |
| Príprava dát | minimálna správnosť na testovacích dátach | minimálny záchyt (0 = bez obmedzenia) | správnosť splnená, ale záchyt nie |
| Rekurentná neurónová sieť | najvyššia chyba (MSE) | minimálny počet zapamätaných predchádzajúcich vstupov (0 = bez obmedzenia) | chyba splnená, ale sieť si pamätá príliš málo vstupov |

---

*Táto príručka vznikla v rámci projektu Erasmus+ AI for VET a je voľne dostupná ako otvorený vzdelávací zdroj (OER) pre všetkých učiteľov a vzdelávacie inštitúcie.*
