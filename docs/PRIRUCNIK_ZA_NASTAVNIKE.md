# AI for VET – Pedagoško-tehnički priručnik za nastavnike
## Integracija interaktivnih simulacija veštačke inteligencije i mašinskog učenja u Moodle okruženje

*Projekat: **AI for VET** (Artificial Intelligence for Vocational Education and Training)*  
*Program: **Erasmus+***  
*Zvanični portal:* [https://aiforvet.eu/](https://aiforvet.eu/)  
*Autori i tehnički tim projekta AI for VET*  
*Verzija priručnika: 1.0 (Oktobar 2026)*  
*Kompatibilnost: Moodle 4.5 LTS, 5.0, 5.1, 5.2, 5.3+*

---

## Sadržaj
1. [Uvod i pedagoški okvir projekta AI for VET](#1-uvod-i-pedagoški-okvir-projekta-ai-for-vet)
   - 1.1. Izazov: Zašto AI u srednjem stručnom obrazovanju?
   - 1.2. Pedagoški pristup: Od vizuelne intuicije do inženjerskog razumevanja
   - 1.3. Arhitektura Moodle AI paketa (Filter, Kviz, TinyMCE editor)
2. [Pedagoški vodič kroz 5 simulacija](#2-pedagoški-vodič-kroz-5-simulacija)
   - 2.1. Modul 1: Gradijentni spust i optimizacija modela (`gradient_descent`)
   - 2.2. Modul 2: Matrica konfuzije i prag odlučivanja (`confusion_matrix`)
   - 2.3. Modul 3: Polinomska regresija i regularizacija (`regression`)
   - 2.4. Modul 4: Nelinearne granice odlučivanja – k-NN i Stablo (`knn_tree`)
   - 2.5. Modul 5: Nenadgledano učenje i K-Means klasterovanje (`kmeans`)
3. [Tehničko uputstvo za nastavnike](#3-tehničko-uputstvo-za-nastavnike)
   - 3.1. Korišćenje vizuelnog TinyMCE editora (`tiny_aisim`)
   - 3.2. Kreiranje zadataka sa automatskim ocenjivanjem (`qtype_aisim`)
   - 3.3. Uvoz gotove baze pitanja (Moodle XML)
4. [Metodičke preporuke i organizacija nastavnog časa](#4-metodičke-preporuke-i-organizacija-nastavnog-časa)
   - 4.1. Struktura radioničkog/blok časa (90 minuta)
   - 4.2. Formativno ocenjivanje i učenje kroz greške
   - 4.3. Diferencirana nastava: Prilagođavanje učenicima različitih nivoa
5. [Prilozi i brzi pregled formula](#5-prilozi-i-brzi-pregled-formula)

---

## 1. Uvod i pedagoški okvir projekta AI for VET

### 1.1. Izazov: Zašto AI u srednjem stručnom obrazovanju?
U savremenoj industriji, elektrotehnici, mašinstvu, mehatronici i informacionim tehnologijama, veštačka inteligencija (AI) i mašinsko učenje (ML) više nisu domen isključivo akademskih istraživanja i doktorskih studija. Učenici srednjih stručnih škola (VET – *Vocational Education and Training*) danas se susreću sa:
- Automatizovanom optičkom kontrolom kvaliteta na proizvodnim trakama.
- Prediktivnim održavanjem elektromotornih pogona na osnovu vibracionih i termičkih senzora.
- Inteligentnim energetskim menadžmentom i optimizacijom potrošnje zgrada.
- Klasifikacijom biomedicinskih signala i autonomnim mobilnim robotima.

Međutim, tradicionalni način podučavanja mašinskog učenja često pati od dve krajnosti:
1. **Preterana matematička apstrakcija:** Uvođenje teških diferencijalnih jednačina, parcijalnih izvoda i matričnih transformacija koje obeshrabruju učenike stručnih škola.
2. **"Crna kutija" biblioteke:** Pozivanje gotovih Python funkcija (`model.fit()`, `model.predict()`) gde učenici vide rezultat, ali nemaju nikakav intuitivni uvid u to šta algoritam zapravo radi "ispod haube".

### 1.2. Pedagoški pristup: Od vizuelne intuicije do inženjerskog razumevanja
Paket alata **AI for VET** uvodi **interaktivno eksperimentalno učenje (Active Inquiry-Based Learning)**:
- **Eksperimentisanje u realnom vremenu:** Učenik menja hiperparametre (klizačima i dugmićima) i direktno na platnu u browseru vidi geometrijski i grafički odziv algoritma u 60 FPS.
- **Klijentsko izvršavanje bez instalacije:** Nema potrebe za instaliranjem Pythona, CUDA drajvera ili podešavanjem Jupyter servera – sve simulacije rade 100% u browseru unutar Moodle-a, čak i na skromnim školskim računarima, tabletima i pametnim telefonima.
- **Dvosmerni pedagoški ciklus:** 
  1. *Faza istraživanja (Istraživačka lekcija):* Učenik slobodno pomera parametre i posmatra zakonitosti.
  2. *Faza provere i primene (Kviz):* Učenik dobija inženjerski problem iz prakse gde algoritam mora dovesti u traženu zonu tolerancije, pri čemu Moodle automatski validira rešenje.

### 1.3. Arhitektura Moodle AI paketa
Komplet se sastoji od tri međusobno povezane komponente:
```
+---------------------------------------------------------------------------------+
|                               MOODLE PLATFORMA                                  |
|                                                                                 |
|  +--------------------------+  +----------------------+  +-------------------+  |
|  |     moodle-tiny_aisim    |  |  moodle-filter_aisim |  | moodle-qtype_aisim|  |
|  |   (TinyMCE 6/7 Editor)   |  | (Rendering & Engine) |  |  (Kviz zadaci &   |  |
|  |                          |  |                      |  | Auto-ocenjivanje) |  |
|  | - Dugme u traci editora  |  | - 5 simulacija na    |  | - Ciljne metrike  |  |
|  | - Modalni dijalog        |  |   HTML5 Canvas-u     |  | - Tolerancije     |  |
|  | - Odabir scenarija       |  | - 60 FPS animacije   |  | - Automatski      |  |
|  | - Dvoklik za izmenu      |  | - Režim samo-čitanje |  |   bodovi i saveti |  |
|  +--------------------------+  +----------------------+  +-------------------+  |
+---------------------------------------------------------------------------------+
```

---

## 2. Pedagoški vodič kroz 5 simulacija

### 2.1. Modul 1: Gradijentni spust i optimizacija modela (`gradient_descent`)
- **Fokus:** Minimizacija funkcije troška $J(w)$, stopa učenja $\alpha$, konveksne vs. nekonveksne funkcije, lokalni minimumi.

#### Pedagoška metafora:
Zamislite planinara koji se po gustoj magli spušta sa vrha planine u dolinu. On ne vidi celu mapu, već samo nagib tla pod svojim nogama. U svakom koraku opipava u kom pravcu je najveći pad i pravi korak u tom smeru. Veličina njegovog koraka jeste **stopa učenja ($\alpha$)**.

#### Inženjerska primena:
- Automatsko podešavanje regulatora u mehatroničkim sistemima.
- Ažuriranje sinaptičkih težina u veštačkim neuronskim mrežama (Backpropagation).

#### Šta učenici uočavaju u simulaciji:
1. **Premala stopa učenja ($\alpha = 0.01$):** Algoritam izuzetno sporo napreduje; posle 50 iteracija je jedva prešao trećinu puta.
2. **Optimalna stopa učenja ($\alpha = 0.10 - 0.20$):** Brza i stabilna konvergencija direktno u minimum funkcije.
3. **Prevelika stopa učenja ($\alpha \ge 1.05$ kod parabole):** Dolazi do **oscilacija i divergencije** – algoritam preskače dno, gubi stabilnost i "odleće" u beskonačnost.
4. **Nekonveksna funkcija troška:** Učenik uočava postojanje lokalnih minimuma i shvata zašto početna tačka ($w_0$) određuje u kojoj "dolini" će model završiti.

#### Tipičan zadatak za kviz:
> *"Ispitujete ponašanje gradijentnog spusta na konveksnoj funkciji troška čiji je minimum $J(w) = 1.00$. Podesite stopu učenja ($\alpha$) i pokrenite algoritam tako da trošak padne na $J(w) \le 1.01$."*

---

### 2.2. Modul 2: Matrica konfuzije i prag odlučivanja (`confusion_matrix`)
- **Fokus:** Evaluacija binarnog klasifikatora, nebalansirane klase, prag odluke $\theta$, TP, FP, FN, TN, Precision, Recall, F1-score.

#### Pedagoški izazov: Zašto tačnost (Accuracy) laže?
U industrijskoj kontroli kvaliteta, recimo da fabrika proizvodi 1000 komada na sat, od kojih je 980 ispravnih i 20 neispravnih (škart). Ako model sve komade proglasi ispravnim, njegova tačnost je neverovatnih **98% (980/1000)**! Međutim, kao industrijski sistem on je **potpuno beskoristan** jer je propustio 100% neispravnih komada.

#### Značenje metrika na jednostavan način:
- **Prag odluke ($\theta$):** Granična verovatnoća iznad koje komad proglašavamo škartom.
- **Precision (Preciznost):** Od svih komada koje je model proglasio škartom, koliko je stvarno bilo neispravno? (Visoka preciznost znači malo nepotrebno bačenih ispravnih komada).
- **Recall (Odziv / Osetljivost):** Od svih stvarno neispravnih komada, koliko je model uspeo da ulovi? (Visok odziv znači da nijedan defektan komad ne ide kupcu).
- **F1-score:** Harmonijska sredina preciznosti i odziva – ključna kompromisna inženjerska metrika.

#### Šta učenici uočavaju u simulaciji:
- Pomeranjem praga $\theta$ ulevo (nizak prag, npr. 0.20): Model postaje "oprezan", hvata sve defekte (Recall raste ka 100%), ali pravi lažne uzbune (Precision pada).
- Pomeranjem praga $\theta$ udesno (visok prag, npr. 0.80): Model je "siguran" kad nešto označi kao škart (Precision raste), ali propušta defekte (Recall drastično pada).

#### Tipičan zadatak za kviz:
> *"Podesite prag odlučivanja ($\theta$) klasifikatora tako da uhvatite većinu škarta bez previše lažnih uzbuna: odziv (Recall) $\ge 0.80$ i preciznost (Precision) $\ge 0.70$."*

---

### 2.3. Modul 3: Polinomska regresija i regularizacija (`regression`)
- **Fokus:** Modelovanje nelinearnih zavisnosti, stepen polinoma $d$, prepodešavanje (*overfitting*), podučenost (*underfitting*), greška na trening vs. test skupu, $L_2$ regularizacija ($\lambda$).

#### Pedagoška metafora:
Podučenost (*underfitting*) je kada učenik pročita samo naslove lekcija i na ispitu ne zna detalje (pravac pokušava da opiše talas). Prepodešavanje (*overfitting*) je kada učenik nabuba napamet svako slovo iz zbirke uključujući štamparske greške – na ispitu čim se promeni jedan broj u zadatku dobija nulu jer nije shvatio suštinu trenda.

#### Inženjerska primena:
- Kalibracija senzora temperature i pritiska.
- Predviđanje opterećenja transformatora i kablova u zavisnosti od doba dana i temperature okoline.

#### Šta učenici uočavaju u simulaciji:
1. **$d = 1$ (Linearni model):** Prava linija ne može da prati sinusoidu ($R^2$ nizak i na trening i na test skupu).
2. **$d = 3$ (Kubni model):** Model glatko prati talas, odlična generalizacija na novim podacima.
3. **$d = 5$ i $\lambda = 0$ (Overfitting):** Kriva prolazi kroz svaku tačku trening skupa sa ogromnim talasima i oscilacijama na krajevima. Greška na treningu je nula, ali na testu je katastrofalna!
4. **Uvođenje regularizacije ($\lambda = 0.01 - 0.10$):** Čak i sa visokim stepenom $d=5$, kazneni član $\lambda \sum w^2$ "pegla" krivu i sprečava preterano podivljale koeficijente.

#### Tipičan zadatak za kviz:
> *"Za periodični signal sa šumom odaberite stepen polinoma najviše 7 tako da model generalizuje trend bez preprilagođavanja, uz validacionu grešku $MSE \le 0.19$."*

---

### 2.4. Modul 4: Nelinearne granice odlučivanja – k-NN i Stablo (`knn_tree`)
- **Fokus:** Nadgledana klasifikacija, $k$-najbližih suseda, stablo odlučivanja, granica odlučivanja (*decision boundary*), hiperparametri $k$ i maksimalna dubina stabla (*max depth*).

#### Poređenje dva pristupa:
- **k-NN (Algoritam baziran na pamćenju instanci):** "Kaži mi ko su ti prijatelji i reći ću ti ko si." Za novu tačku pronalazi $k$ prostorno najbližih komšija i glasa po većini.
- **Stablo odlučivanja (Hijerarhijsko deljenje prostora):** Niz "IF-THEN-ELSE" pitanja koja seku ravan ortogonalnim linijama paralelnim koordinatnim osama.

#### Šta učenici uočavaju u simulaciji:
- **Kod k-NN:** 
  - $k=1$: Granica je izlomljena i ima izolovana "ostrva" oko svake tačke sa šumom (overfitting).
  - $k=3-5$: Glatka, organska granica prilagođena nelinearnim oblicima (npr. Dva polumeseca).
  - $k=15$: Preterano uprošćena granica koja gubi fine detalje.
- **Kod Stabla odlučivanja:**
  - Dubina 1 (Stump): Samo jedan pravougaoni rez – model je previše primitivan.
  - Dubina 3-4: Pravougaona particija koja lepo aproksimira klastere.
  - Dubina 6: Mikroskopski mali pravougaonici oko svake tačke.

---

### 2.5. Modul 5: Nenadgledano učenje i K-Means klasterovanje (`kmeans`)
- **Fokus:** Klasterovanje bez labela, centroidi, Euklidsko rastojanje, inercija (suma kvadrata odstupanja), konvergencija u koracima.

#### Inženjerska primena:
- Segmentacija profila potrošnje električne energije (industrijski, stambeni, noćni potrošači).
- Kompresija boja i analiza satelitskih snimaka poljoprivrednog zemljišta.

#### Šta učenici uočavaju u simulaciji:
1. Učenik klikće na dugme **"Korak (Step)"** i posmatra dva naizmenična koraka:
   - *Faza dodele:* Svaka tačka se boji bojom onog centroida kome je geometrijski najbliža.
   - *Faza ažuriranja:* Svaki centroid se pomera tačno u matematičko težište (srednju vrednost) svojih dodeljenih tačaka.
2. Ako izaberemo $K=2$ na skupu od 3 klastera, vidimo da jedan centroid pokušava da "pokrije" dve odvojene grupe.
3. Posmatranje pada funkcije inercije dok se centroidi ne umire (konvergencija).

---

## 3. Tehničko uputstvo za nastavnike

### 3.1. Korišćenje vizuelnog TinyMCE editora (`tiny_aisim`)
Prilikom izrade lekcije (Page), knjige (Book), forum poruke ili opisa zadatka u Moodle-u:
1. U liniji sa alatkama TinyMCE editora kliknite na **ikonu sa simbolom AI simulacije** (ili u meniju izaberite *Insert > AI Simulation...*).
2. Otvara se modalni prozor sa opcijama:
   - **Tip simulacije:** Izaberite jednu od 5 simulacija iz padajućeg menija.
   - **Specifični parametri:** Izaberite početni scenario, prag, stopu učenja, stepen polinoma ili broj klastera.
   - **Režim samo za čitanje (Read-only):** Označite ovu opciju ukoliko želite da učenici mogu da posmatraju simulaciju, ali ne i da menjaju njene početne parametre.
   - **Pregled uživo (Preview):** Na dnu dijaloga odmah vidite kako će izgledati kartica simulacije.
3. Kliknite na **"Ubaci simulaciju"**.
4. U editoru se pojavljuje zaštićena kartica sa oznakom simulacije.
5. **Izmena postojećeg bloka:** Dovoljno je da u editoru napravite **dvoklik (double-click)** na bilo koju postojeću karticu simulacije – editor će automatski otvoriti prozor sa učitanim parametrima te simulacije!

---

### 3.2. Kreiranje zadataka sa automatskim ocenjivanjem (`qtype_aisim`)
Ako želite da u Moodle kviz ubacite pitanje sa AI simulacijom:
1. Otvorite svoj kviz i izaberite **Pitanja > Dodaj novo pitanje**.
2. Na listi tipova pitanja izaberite **AI Simulation (`aisim`)**.
3. Popunite osnovna polja:
   - **Naziv pitanja:** Npr. *Zadatak: Podešavanje praga za kontrolu kvaliteta*.
   - **Tekst pitanja:** Opišite inženjerski problem koji učenik treba da reši.
4. U sekciji **Postavke AI simulacije**:
   - **Tip simulacije:** Izaberite npr. *Matrica konfuzije*.
   - **Scenario / Skup podataka:** Izaberite npr. *Industrijska kontrola (quality)*.
   - **Ciljna metrika 1 i 2:** Npr. za matricu konfuzije odziv `0.80` i preciznost `0.70`. Značenje polja zavisi od simulacije – pogledajte pomoć (?) uz polja i tabelu u Prilogu.
   - **Dozvoljena tolerancija:** Koristi se samo za gradijentni spust (pola poena za trošak do cilja + tolerancije), npr. `0.05`.
   - **Početna konfiguracija:** Opcioni JSON sa inicijalnim stanjem.
5. Podesite povratne informacije za tačan, delimično tačan i netačan odgovor.
6. Sačuvajte pitanje. Kada učenik pokrene kviz, Moodle prikazuje interaktivno platno. Čim učenik klikne na *"Proveri odgovor"* ili preda test, Moodle upoređuje postignute vrednosti sa zadatim kriterijumima i automatski dodeljuje poene!

---

### 3.3. Uvoz gotove baze pitanja (Moodle XML)
U repozitorijumu `moodle-qtype_aisim` nalazi se gotov pripremljen paket sa reprezentativnim zadacima za svih 5 simulacija:
1. Preuzmite fajl [`questions_aiforvet.xml`](file:///C:/tsp/moodle-qtype_aisim/demo/questions_aiforvet.xml).
2. U vašem Moodle kursu kliknite na **Banka pitanja (Question bank) > Uvoz (Import)**.
3. Kao format fajla izaberite **Moodle XML format**.
4. Prevucite fajl `questions_aiforvet.xml` i kliknite na **Uvezi**.
5. Moodle će uvesti novu kategoriju *"AI za VET - Simulacije"* sa svih 5 kompletno konfigurisanih pitanja koja odmah možete dodati u bilo koji test.

---

## 4. Metodičke preporuke i organizacija nastavnog časa

### 4.1. Struktura radioničkog / blok časa (90 minuta)
Za postizanje maksimalnog učinka preporučuje se struktura od pet koraka:

```
+-----------------------------------------------------------------------------+
| 1. Uvod i inženjerski problem (10-15 min)                                   |
|    - Nastavnik predstavlja stvarni problem iz fabrike/prakse.               |
|    - Postavlja se ključno pitanje: "Kako mašina prepoznaje defekt?"         |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 2. Vođeno vizuelno istraživanje (20 min)                                    |
|    - Učenici otvaraju lekciju u Moodle-u i istražuju simulaciju.             |
|    - Nastavnik zadaje mini-izazove: "Šta se dešava kad stopu stavite na 1.2?"|
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 3. Diskusija u parovima i konceptualizacija (15 min)                        |
|    - Učenici porede zapažanja, definišu ključne pojmove (overfitting, F1).   |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 4. Samostalno rešavanje izazova u Kvizu (25 min)                            |
|    - Učenici rešavaju zadatke sa automatskim ocenjivanjem u Moodle kvizu.   |
|    - Kviz je podešen na "Immediate feedback" sa neograničenim pokušajima.    |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 5. Refleksija i povezivanje sa strukom (15 min)                             |
|    - Zaključak: Kako se ovi modeli primenjuju u njihovom budućem poslu.     |
+-----------------------------------------------------------------------------+
```

### 4.2. Formativno ocenjivanje i učenje kroz greške
- U Moodle kvizu preporučuje se podešavanje ponašanja pitanja na **"Interaktivno sa višestrukim pokušajima" (Interactive with multiple tries)** ili **"Trenutna povratna informacija" (Immediate feedback)**.
- Cilj nije kazniti učenika ako prvi put podesi pogrešan prag, već omogućiti mu da kroz povratnu informaciju zaključi u kom smeru treba da koriguje parametre.
- Učenici razvijaju **inženjersku intuiciju**: sistematski menjaju jednu po jednu varijablu i beleže promenu na izlazu.

### 4.3. Diferencirana nastava: Prilagođavanje učenicima različitih nivoa
- **Osnovni nivo:** Učenici se fokusiraju na pomeranje praga $\theta$ i stope $\alpha$ na pripremljenim 2D skupovima, posmatrajući samo krajnji status (Zeleno = Tačno / Crveno = Netačno).
- **Napredni nivo:** Učenici analiziraju kompromis Precision vs. Recall, posmatraju ponašanje $L_2$ regularizacije kod polinoma petog stepena i eksperimentišu sa nekonveksnim funkcijama sa više lokalnih minimuma.

---

## 5. Prilozi i brzi pregled formula

| Naziv simulacije | Ključni parametri | Matematički model / Formula | Tipičan kriterijum uspeha |
| :--- | :--- | :--- | :--- |
| **Gradijentni spust** | $\alpha$ (stopa), $w_0$ (početak) | $w \leftarrow w - \alpha \nabla J(w)$ | $J(w) \le 1.01$ (minimum je 1.00) |
| **Matrica konfuzije** | $\theta$ (prag odluke) | $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Recall $\ge 0.80$ i Precision $\ge 0.70$ |
| **Polinomska regresija** | $d$ (stepen), $\lambda$ (regularizacija) | $J(w) = \frac{1}{2m}\sum (h_w(x) - y)^2 + \frac{\lambda}{2}\sum w_j^2$ | Validaciona $MSE \le 0.19$ uz $d \le 7$ |
| **k-NN klasifikacija** | $k$ (broj suseda) | $\hat{y} = \arg\max_c \sum_{i \in N_k(x)} I(y_i = c)$ | Tačnost $\ge 98\%$ |
| **K-Means klasterovanje** | $K$ (broj klastera) | $\min_{\mu} \sum_{i=1}^m \sum_{k=1}^K r_{ik} \|x^{(i)} - \mu_k\|^2$ | Konvergencija uz $K = 3$ i $WCSS \le 10$ |

---

*Ovaj priručnik je razvijen u okviru Erasmus+ projekta AI for VET i distribuira se pod slobodnom licencom za sve nastavnike i obrazovne institucije.*
