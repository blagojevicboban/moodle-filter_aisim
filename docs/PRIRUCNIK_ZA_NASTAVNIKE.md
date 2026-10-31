# AI for VET – Pedagoško-tehnički priručnik za nastavnike
## Integracija interaktivnih simulacija veštačke inteligencije i mašinskog učenja u Moodle okruženje

*Projekat: **AI for VET** (Artificial Intelligence for Vocational Education and Training)*  
*Program: **Erasmus+***  
*Zvanični portal:* [https://aiforvet.eu/](https://aiforvet.eu/)  
*Simulacije bez Moodle-a:* [https://sim.aiforvet.eu/](https://sim.aiforvet.eu/)  
*Autori i tehnički tim projekta AI for VET*  
*Verzija priručnika: 1.1 (Oktobar 2026)*  
*Kompatibilnost: Moodle 4.5 LTS, 5.0, 5.1, 5.2, 5.3+*

---

## Sadržaj
1. [Uvod i pedagoški okvir projekta AI for VET](#1-uvod-i-pedagoški-okvir-projekta-ai-for-vet)
   - 1.1. Izazov: Zašto AI u srednjem stručnom obrazovanju?
   - 1.2. Pedagoški pristup: Od vizuelne intuicije do inženjerskog razumevanja
   - 1.3. Arhitektura Moodle AI paketa (Filter, Kviz, TinyMCE editor)
2. [Pedagoški vodič kroz 9 simulacija](#2-pedagoški-vodič-kroz-9-simulacija)
   - 2.1. Modul 1: Gradijentni spust i optimizacija modela (`gradient_descent`)
   - 2.2. Modul 2: Matrica konfuzije i prag odlučivanja (`confusion_matrix`)
   - 2.3. Modul 3: Polinomska regresija i regularizacija (`regression`)
   - 2.4. Modul 4: Nelinearne granice odlučivanja – k-NN i Stablo (`knn_tree`)
   - 2.5. Modul 5: Nenadgledano učenje i K-Means klasterovanje (`kmeans`)
   - 2.6. Modul 6: Perceptron i neuronska mreža (`neural_network`)
   - 2.7. Modul 7: Konvolucija i filteri slike (`convolution`)
   - 2.8. Modul 8: Učenje potkrepljivanjem – Q-learning (`q_learning`)
   - 2.9. Modul 9: Podela podataka i pristrasnost (`data_bias`)
3. [Tehničko uputstvo za nastavnike](#3-tehničko-uputstvo-za-nastavnike)
   - 3.1. Korišćenje vizuelnog TinyMCE editora (`tiny_aisim`)
   - 3.2. Kreiranje zadataka sa automatskim ocenjivanjem (`qtype_aisim`)
   - 3.3. Uvoz gotove baze pitanja (Moodle XML)
   - 3.4. AISim Lab: simulacije bez Moodle-a
4. [Metodičke preporuke i organizacija nastavnog časa](#4-metodičke-preporuke-i-organizacija-nastavnog-časa)
   - 4.1. Struktura radioničkog/blok časa (90 minuta)
   - 4.2. Formativno ocenjivanje i učenje kroz greške
   - 4.3. Diferencirana nastava: Prilagođavanje učenicima različitih nivoa
   - 4.4. Predlog redosleda simulacija
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
- **Eksperimentisanje u realnom vremenu:** Učenik menja hiperparametre (klizačima i dugmićima) i odmah na ekranu vidi kako algoritam reaguje.
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
|  | - Dugme u traci editora  |  | - 9 simulacija u     |  | - Ciljne metrike  |  |
|  | - Modalni dijalog        |  |   browseru           |  | - Tolerancije     |  |
|  | - Odabir scenarija       |  | - 5 jezika           |  | - Automatski      |  |
|  | - Dvoklik za izmenu      |  | - Režim samo-čitanje |  |   bodovi i saveti |  |
|  +--------------------------+  +----------------------+  +-------------------+  |
+---------------------------------------------------------------------------------+
```

Simulacije prate jezik korisnika u Moodle-u: uključeni su engleski, srpski (latinica i ćirilica), slovački i portugalski.

---

## 2. Pedagoški vodič kroz 9 simulacija

Simulacije su poređane od jednostavnijih ka složenijim. Prvih pet pokriva osnove (optimizacija, ocenjivanje modela, prepodešavanje, klasifikacija, klasterovanje), a moduli 6–9 nadograđuju ih neuronskim mrežama, obradom slike, učenjem iz nagrada i pitanjem odgovornog rada sa podacima.

### 2.1. Modul 1: Gradijentni spust i optimizacija modela (`gradient_descent`)
- **Fokus:** Minimizacija funkcije troška J(w), stopa učenja α, konveksne vs. nekonveksne funkcije, lokalni minimumi.

#### Pedagoška metafora:
Zamislite planinara koji se po gustoj magli spušta sa vrha planine u dolinu. On ne vidi celu mapu, već samo nagib tla pod svojim nogama. U svakom koraku opipava u kom pravcu je najveći pad i pravi korak u tom smeru. Veličina njegovog koraka jeste **stopa učenja (α)**.

#### Inženjerska primena:
- Automatsko podešavanje regulatora u mehatroničkim sistemima.
- Ažuriranje sinaptičkih težina u veštačkim neuronskim mrežama (Backpropagation).

#### Šta učenici uočavaju u simulaciji:
1. **Premala stopa učenja (α = 0.01):** Algoritam izuzetno sporo napreduje; posle 50 iteracija je jedva prešao trećinu puta.
2. **Optimalna stopa učenja (α = 0.10 – 0.20):** Brza i stabilna konvergencija direktno u minimum funkcije.
3. **Prevelika stopa učenja (α ≥ 1.05 kod parabole):** Dolazi do **oscilacija i divergencije** – algoritam preskače dno, gubi stabilnost i "odleće" u beskonačnost.
4. **Nekonveksna funkcija troška:** Učenik uočava postojanje lokalnih minimuma i shvata zašto početna tačka (w₀) određuje u kojoj "dolini" će model završiti.

#### Tipičan zadatak za kviz:
> *"Ispitujete ponašanje gradijentnog spusta na konveksnoj funkciji troška čiji je minimum J(w) = 1.00. Podesite stopu učenja (α) i pokrenite algoritam tako da trošak padne na J(w) ≤ 1.01."*

---

### 2.2. Modul 2: Matrica konfuzije i prag odlučivanja (`confusion_matrix`)
- **Fokus:** Evaluacija binarnog klasifikatora, nebalansirane klase, prag odluke θ, TP, FP, FN, TN, Precision, Recall, F1-score.

#### Pedagoški izazov: Zašto tačnost (Accuracy) laže?
U industrijskoj kontroli kvaliteta, recimo da fabrika proizvodi 1000 komada na sat, od kojih je 980 ispravnih i 20 neispravnih (škart). Ako model sve komade proglasi ispravnim, njegova tačnost je neverovatnih **98% (980/1000)**! Međutim, kao industrijski sistem on je **potpuno beskoristan** jer je propustio 100% neispravnih komada.

#### Značenje metrika na jednostavan način:
- **Prag odluke (θ):** Granična verovatnoća iznad koje komad proglašavamo škartom.
- **Precision (Preciznost):** Od svih komada koje je model proglasio škartom, koliko je stvarno bilo neispravno? (Visoka preciznost znači malo nepotrebno bačenih ispravnih komada).
- **Recall (Odziv / Osetljivost):** Od svih stvarno neispravnih komada, koliko je model uspeo da ulovi? (Visok odziv znači da nijedan defektan komad ne ide kupcu).
- **F1-score:** Harmonijska sredina preciznosti i odziva – ključna kompromisna inženjerska metrika.

#### Šta učenici uočavaju u simulaciji:
- Pomeranjem praga θ ulevo (nizak prag, npr. 0.20): Model postaje "oprezan", hvata sve defekte (Recall raste ka 100%), ali pravi lažne uzbune (Precision pada).
- Pomeranjem praga θ udesno (visok prag, npr. 0.80): Model je "siguran" kad nešto označi kao škart (Precision raste), ali propušta defekte (Recall drastično pada).

#### Tipičan zadatak za kviz:
> *"Podesite prag odlučivanja (θ) klasifikatora tako da uhvatite većinu škarta bez previše lažnih uzbuna: odziv (Recall) ≥ 0.80 i preciznost (Precision) ≥ 0.70."*

---

### 2.3. Modul 3: Polinomska regresija i regularizacija (`regression`)
- **Fokus:** Modelovanje nelinearnih zavisnosti, stepen polinoma d, prepodešavanje (*overfitting*), podučenost (*underfitting*), greška na trening vs. test skupu, L₂ regularizacija (λ).

#### Pedagoška metafora:
Podučenost (*underfitting*) je kada učenik pročita samo naslove lekcija i na ispitu ne zna detalje (pravac pokušava da opiše talas). Prepodešavanje (*overfitting*) je kada učenik nabuba napamet svako slovo iz zbirke uključujući štamparske greške – na ispitu čim se promeni jedan broj u zadatku dobija nulu jer nije shvatio suštinu trenda.

#### Inženjerska primena:
- Kalibracija senzora temperature i pritiska.
- Predviđanje opterećenja transformatora i kablova u zavisnosti od doba dana i temperature okoline.

#### Šta učenici uočavaju u simulaciji:
1. **d = 1 (Linearni model):** Prava linija ne može da prati sinusoidu (R² nizak i na trening i na test skupu).
2. **d = 3 (Kubni model):** Model glatko prati talas, odlična generalizacija na novim podacima.
3. **d = 5 i λ = 0 (Overfitting):** Kriva prolazi kroz svaku tačku trening skupa sa ogromnim talasima i oscilacijama na krajevima. Greška na treningu je nula, ali na testu je katastrofalna!
4. **Uvođenje regularizacije (λ = 0.01 – 0.10):** Čak i sa visokim stepenom d = 5, kazneni član λ·Σw² "pegla" krivu i sprečava preterano podivljale koeficijente.

#### Tipičan zadatak za kviz:
> *"Za periodični signal sa šumom odaberite stepen polinoma najviše 7 tako da model generalizuje trend bez preprilagođavanja, uz validacionu grešku MSE ≤ 0.19."*

---

### 2.4. Modul 4: Nelinearne granice odlučivanja – k-NN i Stablo (`knn_tree`)
- **Fokus:** Nadgledana klasifikacija, k-najbližih suseda, stablo odlučivanja, granica odlučivanja (*decision boundary*), hiperparametri k i maksimalna dubina stabla (*max depth*).

#### Poređenje dva pristupa:
- **k-NN (Algoritam baziran na pamćenju instanci):** "Kaži mi ko su ti prijatelji i reći ću ti ko si." Za novu tačku pronalazi k prostorno najbližih komšija i glasa po većini.
- **Stablo odlučivanja (Hijerarhijsko deljenje prostora):** Niz "IF-THEN-ELSE" pitanja koja seku ravan ortogonalnim linijama paralelnim koordinatnim osama.

#### Šta učenici uočavaju u simulaciji:
- **Kod k-NN:** 
  - k = 1: Granica je izlomljena i ima izolovana "ostrva" oko svake tačke sa šumom (overfitting).
  - k = 3 – 5: Glatka, organska granica prilagođena nelinearnim oblicima (npr. Dva polumeseca).
  - k = 15: Preterano uprošćena granica koja gubi fine detalje.
- **Kod Stabla odlučivanja:**
  - Dubina 1 (Stump): Samo jedan pravougaoni rez – model je previše primitivan.
  - Dubina 3 – 4: Pravougaona particija koja lepo aproksimira klastere.
  - Dubina 6: Mikroskopski mali pravougaonici oko svake tačke.

---

### 2.5. Modul 5: Nenadgledano učenje i K-Means klasterovanje (`kmeans`)
- **Fokus:** Klasterovanje bez labela, centroidi, Euklidsko rastojanje, inercija (suma kvadrata odstupanja), konvergencija u koracima.

#### Inženjerska primena:
- Segmentacija profila potrošnje električne energije (industrijski, stambeni, noćni potrošači).
- Kompresija boja i analiza satelitskih snimaka poljoprivrednog zemljišta.

#### Šta učenici uočavaju u simulaciji:
1. Učenik klikće na dugme **"Sledeći korak"** i posmatra dva naizmenična koraka:
   - *Faza dodele:* Svaka tačka se boji bojom onog centroida kome je geometrijski najbliža.
   - *Faza ažuriranja:* Svaki centroid se pomera tačno u matematičko težište (srednju vrednost) svojih dodeljenih tačaka.
2. Ako izaberemo K = 2 na skupu od 3 klastera, vidimo da jedan centroid pokušava da "pokrije" dve odvojene grupe.
3. Posmatranje pada funkcije inercije dok se centroidi ne umire (konvergencija).

---

### 2.6. Modul 6: Perceptron i neuronska mreža (`neural_network`)
- **Fokus:** Perceptron, linearna razdvojivost, skriveni sloj, aktivacione funkcije (sigmoid, tanh, ReLU), epohe, funkcija gubitka (log-loss), stopa učenja, početne težine.

#### Pedagoška metafora:
Jedan neuron je kao lenjir: može da povuče samo jednu pravu liniju i kaže "levo je crveno, desno je plavo". Skriveni sloj je tim od nekoliko lenjira čije linije izlazni neuron spaja u savijenu granicu – kao kad od više pravih ivica isečete krug od papira. Što više neurona, to finije sečenje.

#### Inženjerska primena:
- Prepoznavanje ispravnih i neispravnih delova iz više senzorskih merenja.
- Neuronske mreže su osnova prepoznavanja govora, slike i velikih jezičkih modela – ovde je ista ideja u najmanjem mogućem obliku.

#### Šta učenici uočavaju u simulaciji:
1. **Linearno razdvojivi podaci, 0 skrivenih slojeva:** Perceptron brzo nauči pravu liniju i dostiže skoro 100%.
2. **XOR (ukršteni kvadranti), 0 skrivenih slojeva:** Tačnost ostaje oko 50% bez obzira na trajanje treninga – nijedna prava ne razdvaja klase. Simulacija to i kaže u savetu.
3. **XOR sa jednim skrivenim slojem od 2–4 neurona:** Granica se savija i tačnost prelazi 98%. Debljina linija na crtežu mreže pokazuje koje veze su postale jake.
4. **Krug u krugu:** Sa 2 neurona mreža često zastane; sa 3–4 neurona i dugmetom "Nove težine" pronalazi zatvorenu granicu oko unutrašnje klase.
5. **Prevelika stopa učenja (npr. 3):** Gubitak "eksplodira" – ista pouka kao kod gradijentnog spusta, jer je trening mreže upravo gradijentni spust.

#### Tipičan zadatak za kviz:
> *"Kod podataka XOR perceptron ostaje na 50% tačnosti. Dodajte skriveni sloj i postignite tačnost ≥ 98% sa najviše 4 skrivena neurona."*

---

### 2.7. Modul 7: Konvolucija i filteri slike (`convolution`)
- **Fokus:** Slika kao tabela brojeva, kernel 3×3, delilac, zamućivanje, izoštravanje, detekcija ivica (Sobel, Laplasijan), apsolutna vrednost, konvolucione neuronske mreže (CNN).

#### Pedagoška metafora:
Kernel je mala "lupa sa pravilom" koja klizi preko slike. Na svakom mestu pogleda 9 piksela, pomnoži ih svojim težinama i upiše jedan broj. Ako su sve težine iste, lupa uprosečava (zamućuje). Ako jednu stranu oduzima od druge, lupa "vidi" samo mesta gde se svetlina naglo menja – ivice.

#### Inženjerska primena:
- Optička kontrola kvaliteta: pronalaženje ogrebotina, pukotina i ivica delova na proizvodnoj traci.
- Uklanjanje šuma sa slika kamera i medicinskih snimaka.
- Prvi slojevi svake mreže za prepoznavanje slika (CNN) uče upravo ovakve kernele.

#### Šta učenici uočavaju u simulaciji:
1. **Dodir na piksel** prikazuje celo računanje: 9 vrednosti, 9 težina, zbir i deljenje. Ovo je najvažniji trenutak časa – konvolucija prestaje da bude "magija".
2. **Zamućenje (prosek) i Gausovo zamućenje:** Šum nestaje, ali i ivice postaju mekše.
3. **Zbir težina / delilac ≠ 1:** Cela slika postaje svetlija ili tamnija – simulacija na to upozorava.
4. **Sobel bez apsolutne vrednosti:** Ivice od svetlog ka tamnom daju negativne vrednosti i vide se kao crne. Uz |x| vide se obe vrste ivica.
5. **Ciljna slika i podudaranje (R²):** Učenik pogađa ili sam sastavlja filter koji daje zadati rezultat.

#### Tipičan zadatak za kviz:
> *"Slika je puna šuma, a gotovi filteri su skriveni. Upišite težine kernela i delilac tako da rezultat bude kao na ciljnoj slici, uz podudaranje ≥ 95%."*
>
> (Jedno rešenje: težine 1 2 1 / 2 4 2 / 1 2 1 i delilac 16.)

---

### 2.8. Modul 8: Učenje potkrepljivanjem – Q-learning (`q_learning`)
- **Fokus:** Agent, stanje, akcija, nagrada, Q-tabela, epizoda, stopa učenja α, faktor umanjenja γ, istraživanje ε (*exploration vs. exploitation*), politika.

#### Pedagoška metafora:
Dresura psa: niko psu ne objašnjava pravila, on samo dobija poslasticu kad uradi pravu stvar. Posle mnogo pokušaja pamti koji potez u kojoj situaciji donosi nagradu. Agent u simulaciji radi isto: −0.1 za svaki korak, +10 na cilju, −10 kad padne sa litice.

#### Inženjerska primena:
- Upravljanje robotima i autonomnim vozilima (kretanje, hvatanje predmeta).
- Optimizacija upravljanja energijom (punjenje baterija, grejanje) i semaforima.
- AI za igre – od šaha i Go do video igara.

#### Šta učenici uočavaju u simulaciji:
1. **Lavirint:** Na početku agent luta nasumično. Posle nekoliko desetina epizoda boje polja i strelice formiraju "putokaz" do cilja, a putanja naučene politike se skraćuje na 10 koraka.
2. **Litica, ε veliko (npr. 0.4):** Agent nauči najkraći put tik uz ivicu, ali tokom treninga često nasumično padne – prosečna nagrada je niska. Sa ε = 0.05 pada ređe.
3. **Blizu ili daleko, γ = 0.5:** Agent uzima malo blago T (+3) jer mu budućnost malo vredi. Kada se γ poveća (npr. 0.9), veliki cilj G (+10) postaje vredniji – ali samo ako ga je agent uopšte otkrio, za šta treba dovoljno ε.
4. **Dodir na polje** prikazuje Q-vrednosti sva četiri poteza – učenik vidi brojeve iza strelice.

#### Tipičan zadatak za kviz:
> *"Agent bira malo blago T (+3) u blizini. Promenite faktor umanjenja γ i istraživanje ε tako da nauči da ide do velikog cilja G (+10) u 4 koraka, za najviše 1000 epizoda. Objasnite ulogu svakog parametra."*

---

### 2.9. Modul 9: Podela podataka i pristrasnost (`data_bias`)
- **Fokus:** Skup za trening i test skup, generalizacija, pamćenje (k = 1), reprezentativan uzorak, pristrasnost podataka (*data bias*), merenje tačnosti po grupama, pravednost modela.

#### Pedagoška metafora:
Ako ispit sastavite samo od zadataka koje je učenik već video, odlična ocena ne govori ništa. A ako i vežbe i ispit dolaze iz udžbenika u kome nema nijednog zadatka iz geometrije, učenik će i dalje imati odličnu ocenu – i pasti prvi put kad vidi trougao. Test skup može da otkrije samo greške koje uzorak sadrži.

#### Inženjerska primena:
- Model za kontrolu kvaliteta treniran na delovima iz jedne fabrike, a korišćen u drugoj (druga mašina, drugi materijal).
- Sistemi za odobravanje kredita, zapošljavanje ili medicinsku dijagnostiku koji rade lošije za grupe kojih je bilo malo u podacima.
- Evropski AI Act zahteva upravo ovo: kvalitetne, reprezentativne podatke i proveru za različite grupe.

#### Šta učenici uočavaju u simulaciji:
1. **Udeo fabrike B 5%:** Test skup pokazuje oko 95%, a stvarna tačnost za fabriku B je mnogo niža. Prosečan rezultat sakriva grupu koja je retka.
2. **Povećanje udela B (npr. na 30–50%):** Tačnost za B naglo raste, iako ukupan broj uzoraka ostaje isti – *koji* podaci su važniji od *koliko* podataka.
3. **"B se ponaša slično":** Pristrasnost ne škodi kada se grupe ne razlikuju – problem nastaje kada se razlikuju, a nisu zastupljene.
4. **k = 1:** Tačnost na treningu je 100% (model je zapamtio svaku tačku), ali to ništa ne govori o kvalitetu.
5. **Mali test skup:** Rezultat testa skače sa svakim novim uzorkom (🎲) – mali test je nepouzdano merilo.

#### Tipičan zadatak za kviz:
> *"Prikupljanje podataka je skupo. Sa najviše 120 uzoraka postignite da stvarna tačnost obe fabrike bude ≥ 88%. Šta je važnije: koliko podataka ili koji podaci?"*

---

## 3. Tehničko uputstvo za nastavnike

### 3.1. Korišćenje vizuelnog TinyMCE editora (`tiny_aisim`)
Prilikom izrade lekcije (Page), knjige (Book), forum poruke ili opisa zadatka u Moodle-u:
1. U liniji sa alatkama TinyMCE editora kliknite na **ikonu sa simbolom AI simulacije** (ili u meniju izaberite *Insert > AI Simulation...*).
2. Otvara se modalni prozor sa opcijama:
   - **Tip simulacije:** Izaberite jednu od 9 simulacija iz padajućeg menija.
   - **Specifični parametri:** Izaberite početni scenario, prag, stopu učenja, stepen polinoma, broj klastera, arhitekturu mreže, sliku i filter, mapu agenta ili udeo grupe u podacima.
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
   - **Dozvoljena tolerancija:** Koristi se za gradijentni spust (pola poena za trošak do cilja + tolerancije) i konvoluciju (pola poena za podudaranje do cilja − tolerancije), npr. `0.05`.
   - **Početna konfiguracija:** Opcioni JSON sa inicijalnim stanjem.
5. Podesite povratne informacije za tačan, delimično tačan i netačan odgovor.
6. Sačuvajte pitanje. Kada učenik pokrene kviz, Moodle prikazuje interaktivnu simulaciju. Čim učenik klikne na *"Proveri odgovor"* ili preda test, Moodle upoređuje postignute vrednosti sa zadatim kriterijumima i automatski dodeljuje poene!

---

### 3.3. Uvoz gotove baze pitanja (Moodle XML)
U repozitorijumu `moodle-qtype_aisim` nalazi se gotov paket od 27 zadataka – po tri za svaku od 9 simulacija (osnovni, srednji i napredni nivo):
1. Preuzmite fajl [`questions_aiforvet.xml`](https://github.com/blagojevicboban/moodle-qtype_aisim/raw/master/demo/questions_aiforvet.xml).
2. U vašem Moodle kursu kliknite na **Banka pitanja (Question bank) > Uvoz (Import)**.
3. Kao format fajla izaberite **Moodle XML format**.
4. Prevucite fajl `questions_aiforvet.xml` i kliknite na **Uvezi**.
5. Moodle će uvesti novu kategoriju *"AI za VET - Baza simulacionih zadataka (27 pitanja)"* sa kompletno konfigurisanim pitanjima koja odmah možete dodati u bilo koji test.

---

### 3.4. AISim Lab: simulacije bez Moodle-a
Na adresi [sim.aiforvet.eu](https://sim.aiforvet.eu/) svih 9 simulacija radi samostalno, bez naloga i bez prikupljanja podataka, na srpskom, engleskom, slovačkom i portugalskom. Svaka simulacija ima kratko objašnjenje i tri izazova.
- **Domaći zadatak i rad na telefonu:** Učenici mogu da vežbaju van Moodle-a; napredak u izazovima čuva se samo u njihovom browseru.
- **Dugme "Kopiraj link do ove postavke":** Pravi link do tačno određenog stanja simulacije – zgodno za projektor ili za poruku učenicima.
- **Dugme "Ugradi u Moodle":** Daje gotov HTML kod za tu postavku koji se lepi u bilo koji Moodle tekst (potreban je `filter_aisim`).

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
|    - Učenici otvaraju lekciju u Moodle-u i istražuju simulaciju.            |
|    - Nastavnik zadaje mini-izazove: "Šta se dešava kad stopu stavite na 1.2?"|
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 3. Diskusija u parovima i konceptualizacija (15 min)                        |
|    - Učenici porede zapažanja, definišu ključne pojmove (overfitting, F1).  |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 4. Samostalno rešavanje izazova u Kvizu (25 min)                            |
|    - Učenici rešavaju zadatke sa automatskim ocenjivanjem u Moodle kvizu.   |
|    - Kviz je podešen na "Immediate feedback" sa neograničenim pokušajima.   |
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
- Napredni zadaci iz baze (npr. Q-learning i pristrasnost podataka) završavaju se pitanjem "Objasnite…" – iskoristite ga za kratak usmeni ili pisani odgovor posle kviza.

### 4.3. Diferencirana nastava: Prilagođavanje učenicima različitih nivoa
- **Osnovni nivo:** Učenici se fokusiraju na pomeranje praga θ i stope α na pripremljenim 2D skupovima, posmatrajući samo krajnji status (Zeleno = Tačno / Crveno = Netačno). Kod konvolucije biraju gotove filtere, a kod Q-learninga posmatraju kako agent uči lavirint.
- **Napredni nivo:** Učenici analiziraju kompromis Precision vs. Recall, posmatraju ponašanje L₂ regularizacije kod polinoma petog stepena, ručno sastavljaju kernele, traže najmanju neuronsku mrežu koja rešava zadatak i objašnjavaju uticaj γ i ε na ponašanje agenta.

### 4.4. Predlog redosleda simulacija
| Čas / tema | Simulacije | Ključna ideja |
| :--- | :--- | :--- |
| 1. Kako model uči | Gradijentni spust | Učenje = smanjivanje greške u malim koracima |
| 2. Kako merimo model | Matrica konfuzije, Podela podataka i pristrasnost | Tačnost nije dovoljna; test mora biti pošten i reprezentativan |
| 3. Složenost modela | Polinomska regresija, k-NN i Stablo | Prejednostavan vs. prepodešen model |
| 4. Učenje bez odgovora | K-Means | Grupisanje bez labela |
| 5. Neuronske mreže i slika | Perceptron i neuronska mreža, Konvolucija | Od jednog neurona do prepoznavanja slike |
| 6. Učenje iz nagrada | Q-learning | Istraživanje vs. iskorišćavanje, vrednost budućnosti |

---

## 5. Prilozi i brzi pregled formula

| Naziv simulacije | Ključni parametri | Matematički model / Formula | Tipičan kriterijum uspeha |
| :--- | :--- | :--- | :--- |
| **Gradijentni spust** | α (stopa), w₀ (početak) | w ← w − α · J′(w) | J(w) ≤ 1.01 (minimum je 1.00) |
| **Matrica konfuzije** | θ (prag odluke) | F1 = 2 · P · R / (P + R) | Recall ≥ 0.80 i Precision ≥ 0.70 |
| **Polinomska regresija** | d (stepen), λ (regularizacija) | J(w) = (1/2m) · Σ(h(x) − y)² + (λ/2) · Σw² | Validaciona MSE ≤ 0.19 uz d ≤ 7 |
| **k-NN klasifikacija** | k (broj suseda) | ŷ = najčešća klasa među k najbližih suseda | Tačnost ≥ 98% |
| **K-Means klasterovanje** | K (broj klastera) | min Σ ‖x − μ‖² (zbir kvadrata rastojanja do centroida) | Konvergencija uz K = 3 i WCSS ≤ 10 |
| **Neuronska mreža** | slojevi, neuroni, aktivacija, stopa učenja | a = f(w · x + b); w ← w − α · ∂L/∂w | Tačnost ≥ 98% uz najviše 4 skrivena neurona |
| **Konvolucija** | kernel 3×3, delilac, \|x\| | y = Σ (težina · piksel) / delilac | Podudaranje sa ciljem (R²) ≥ 95% |
| **Q-learning** | α, γ, ε, broj epizoda | Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s′,·) − Q(s,a)) | Najkraći put (npr. 10 koraka) za ≤ 200 epizoda |
| **Podela podataka i pristrasnost** | veličina uzorka, udeo grupe B, test %, k | tačnost najslabije grupe = min(tačnost A, tačnost B) | Obe fabrike ≥ 88% uz ≤ 120 uzoraka |

#### Značenje polja "Ciljna metrika" u pitanjima (`qtype_aisim`)
| Simulacija | Ciljna metrika 1 | Ciljna metrika 2 | Pola poena |
| :--- | :--- | :--- | :--- |
| Gradijentni spust | najveći dozvoljeni trošak J(w) | – | trošak do cilja + tolerancije |
| Matrica konfuzije | najmanji odziv (Recall) | najmanja preciznost (Precision) | ispunjen jedan uslov |
| Polinomska regresija | najveća validaciona MSE | najveći stepen d | ispunjen jedan uslov |
| k-NN i Stablo | najmanja tačnost na testu | – | tačnost do 5% ispod cilja |
| K-Means | najveća inercija (WCSS) | tačan broj klastera K (0 = bilo koji) | ispunjen jedan uslov |
| Neuronska mreža | najmanja tačnost | najviše skrivenih neurona (0 = bez ograničenja) | ispunjen jedan uslov |
| Konvolucija | najmanje podudaranje (R²) | – | podudaranje do cilja − tolerancije |
| Q-learning | najviše koraka do cilja | najviše epizoda treninga (0 = bez ograničenja) | agent stiže do cilja, ali duže ili posle više epizoda |
| Podela podataka i pristrasnost | najmanja tačnost najslabije fabrike | najveći broj uzoraka (0 = bez ograničenja) | tačnost ispunjena, ali uz previše uzoraka |

---

*Ovaj priručnik je razvijen u okviru Erasmus+ projekta AI for VET i distribuira se pod slobodnom licencom za sve nastavnike i obrazovne institucije.*
