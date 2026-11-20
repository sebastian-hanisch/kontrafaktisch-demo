# Was müsste anders sein? Wie lange ein Weg gilt: Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kontrafaktisch-demo.streamlit.app/)** -->

Stück der Linie **„Beitrag und Beleg“** (Feld „Modelle verstehen“, Bereich Lernen) in der Reihe „Konzepte“ der Website „Sebastian Hanisch, Operations Research und Machine Learning“. Die Demo macht aus einem Beitrag eine Handlung: Ein Modell sagt „Langsteher“ (Verweilzeit über 4 Tage), und ein **Weg** ist die **billigste erlaubte Änderung** der Merkmale, nach der es „kein Langsteher“ sagt. Die Annahme, die hier aufgehoben wird, lautet: *„Was das Modell am Zielpunkt sagt, gilt dort auch.“* Gezählt wird, wie viele Wege nach einem Neu-Training des Modells oder in der Wahrheit des Generators nicht mehr gelten.

Die Container sind **ausgedacht**: Art, Kunde, Zoll, Wochenende und Vorlauf (die Zeitspanne in Tagen bis zur Schiffsabfahrt, nur bei Export und Umschlag) bestimmen die Verweilzeit nach einer bekannten Formel; das Merkmal Gewicht ist reines Rauschen. Es gibt keine reale Datenquelle. In den gelesenen Quellen fand sich kein Beleg dafür, dass kontrafaktische Erklärungen in Betrieb oder Logistik bessere Entscheidungen liefern; die Demo behauptet das nicht. Gezählt werden **Instanzen** (bis zu 25 je Zufalls-Seed), **Zufalls-Seeds** (je ein Trainingsdatensatz mit n = 1 000) und **Gruppen** (je 12 getrennte Zufalls-Seeds, also bis zu 300 Instanzen).

## Warum dieses Problem

Eine kontrafaktische Erklärung (Wachter u. a.: kleinste Änderung der Eingabe, die die Vorhersage ändert) liest sich wie ein Rat: „Wäre der Vorlauf kürzer gewesen, wäre der Container kein Langsteher.“ Der billigste Weg liegt wegen der kleinen Kosten oft dicht an der Entscheidungsgrenze des Modells, aber nicht immer unmittelbar auf ihr (Messung im Abschnitt „Befunde“, „Lage des Wegs zur Grenze“). Ändert sich das Modell ein wenig, etwa durch ein Neu-Training mit frischen Daten, kann der Rat falsch werden. Die Demo rechnet das für **fünf Modellklassen** nach, damit die Aussage nicht an einer Klasse hängt: logistische Regression, Baum (scikit-learn, Tiefe 4), Boosting (100 Bäume, Tiefe 3), kleines Netz (16 Einheiten) und den **exakt optimalen Baum** (Tiefe 3, Dynamische Programmierung über ein Raster von 47 Schnitten, in `kfk_tree.py`).

Namen: In der Demo heißt die Methode „kontrafaktische Erklärung“ (im Fließtext „was müsste anders sein“); ein **Weg** ist die Menge der geänderten Merkmale mit ihren Zielwerten. „Rückgriff“ (englisch *algorithmic recourse*) meint in der Literatur etwas Verwandtes (eine Änderung, die der Betroffene tatsächlich umsetzen kann); im Portfolio gehört das Wort zu einer anderen Methode (Korrekturentscheidung der zweiten Stufe bei Optimierung unter Unsicherheit). „Kausal“ im Sinn von Karimi u. a. (Eingriff im Strukturmodell) ist hier nicht gemeint.

## Befunde und Korrekturen gegenüber Plan und Vormessung

Die Zahlen der Vormessung (andere Zufalls-Ziehungen, numpy statt SplitMix64, 4 000 Testzeilen je Zufalls-Seed) sind Erwartungen; im Bau sind sie mit dem Zufall der Demo (SplitMix64, `kfk_rng.py`) neu gemessen (60 Zufalls-Seeds in fünf Gruppen, `tools/kfk_make_fragilitaet.py`). Verschoben haben sich:

| Aussage | Vormessung (Spanne über fünf Gruppen) | Bau (diese Demo) |
|---|---|---|
| Ungültig nach R1, Abstand 0, Politik A: Regression / Baum / Boosting / Netz / exakt optimaler Baum | 17 bis 24 / 19 bis 40 / 27 bis 40 / 26 bis 35 / 12 bis 26 % | 15 bis 22 / 35 bis 44 / 31 bis 35 / 26 bis 33 / 15 bis 40 % |
| Dasselbe, Abstand 1,0 | 0 bis 2 / 7 bis 12 / 9 bis 13 / 3 bis 9 / 7 bis 27 % | 0 / 0 bis 24 / 8 bis 13 / 8 bis 10 / 11 bis 34 % |
| In der Wahrheit ungültig, Abstand 0 / 1,0, exakt optimaler Baum | 27 bis 37 / 17 bis 26 % | 30 bis 36 / 21 bis 27 % |
| Kein Weg, Politik C / B (Spanne über Gruppen und die fünf Klassen) | 46 bis 55 / 28 bis 43 % | 41 bis 55 / 30 bis 45 % |
| Mehr als ein Weg (Mengen geänderter Merkmale, Abstand 0): Regression / Baum / Boosting / Netz / exakt optimaler Baum | 15 bis 23 / 0 bis 5 / 18 bis 33 / 18 bis 29 / 4 bis 12 % | 14 bis 19 / 0 bis 3 / 20 bis 27 / 20 bis 23 / 5 bis 7 % |
| Weg ändert (allein oder neben anderen Merkmalen) das Merkmal ohne Wirkung (Gewicht): Regression / Baum / Boosting / Netz / exakt optimaler Baum | 0,7 bis 1,7 / 0 bis 1,3 / 5,0 bis 8,3 / 2,0 bis 6,3 / 0 bis 1,3 % | 0,3 bis 1,3 / 0 / 4,0 bis 6,3 / 2,7 bis 7,3 / 0 bis 0,7 % |
| Mehr als ein Weg nach Blättern, Schwelle 10 Tage, Tiefe 5 (M5: 86 %, Band 87 bis 96 %) | 87 bis 96 % | 61 bis 87 % |
| Dasselbe, Schwelle 4 Tage, Tiefe 5 | 1 bis 8 % | 2 bis 16 % |

Gründe der Verschiebungen, soweit sie gemessen sind: andere Zufalls-Ziehungen (der Baum hat bei Abstand 1,0 je Gruppe eine Spanne von 0 bis 24 %; vermutlich, weil sich die Instanzen eines Zufalls-Seeds ähneln, gemessen ist das nicht); der Weg wird **nur auf Rasterpunkten** gebildet und mit strukturellen Regeln (M5 bildete ihn als „Quaderrand plus ε“ ohne Regeln); der Baum im Tab „Gleich gute Wege“ wird auf gerundeten Daten trainiert. Ursachen darüber hinaus sind nicht untersucht. **Zweite Messung** mit 60 weiteren Zufalls-Seeds (`--basis 60000 --tab-basis 30000`, Messung dieser README, nicht im Test): ungültig nach R1, Abstand 0: Regression 14 bis 24 % (18,4 % über alle), Baum 25 bis 43 % (37,4), Boosting 29 bis 37 % (32,9), Netz 29 bis 34 % (31,1), exakt optimaler Baum 12 bis 35 % (24,3); Abstand 1,0: 0 bis 1 / 4 bis 11 / 6 bis 11 / 5 bis 9 / 12 bis 32 %. Die Größenordnung bleibt, die Punktzahlen springen: darum steht jede Aussage als Spanne. Weitere Korrekturen: Der Plan nannte „Rückgriff gleich Aufzählung in 360 von 360 Fällen“ als Messung; im Test stehen 108 von 108 (ganzzahliges Programm) und 45 von 45 (Blattquader, je zwei Bäume). Testzeilen je Zufalls-Seed sind 400 statt 4 000 (Genauigkeit also mit größerem Standardfehler, Mittel über 60 Zufalls-Seeds).

## Modell

Annahmen und Formeln stehen im Expander **📐 Mathematische Formulierung** der App (Lesehilfe, vollständiges Modell mit Variablenbereichen, Kosten, Politik, Sicherheitsabstand; das ganzzahlige Programm für das lineare Modell). Kurz: Kandidaten sind die 13 104 strukturell zulässigen Rasterpunkte (Art 4, Kunde 12, Zoll 2, Wochenende 2, Vorlauf 21 Stufen zu 0,5 Tagen, Gewicht 13 Stufen; von 52 416 Kombinationen). **Kosten sind gesetzt:** 1 je geänderte Kategorie oder Binärgröße, Vorlauf |Δ| geteilt durch die Standardabweichung der von 0 verschiedenen Vorlaufzeiten im Training, Gewicht |Δ|. **Politik** A: alles veränderbar, B: Vorlauf und Zoll, C: nur Vorlauf. **Sicherheitsabstand** m: verlangt wird Logit ≤ −m − 10⁻⁶. Bei Gleichstand der Kosten gewinnt der kleinste Kandidaten-Index.

Der **exakt optimale Baum** minimiert den Trainingsfehler über alle Bäume der Tiefe ≤ 3 mit Schnitten aus dem Raster (mindestens 10 Zeilen je Blatt); „exakt“ gilt über dieses Raster. **Gleichstandsregel (Linienregel):** bei mehreren Bäumen mit gleichem Fehler gewinnt der mit dem **kleinsten Index in der festen Aufzählungsreihenfolge** (Vorordnung der Schnittindizes, Blatt = −1); die zweite Regel („größter Index“) zeigt die Spanne. Auf dem Datensatz mit Zufalls-Seed 1 haben beide Bäume denselben Trainingsfehler 220 von 1 000 und verschiedene Gestalt (7 und 8 Blätter).

## Methodik

* **Aufzählung** über alle Kandidaten ist das Hauptverfahren; **ganzzahliges Programm** (`scipy.optimize.milp`, HiGHS) nur für die logistische Regression als zweiter Rechenweg; **Blattquader** für die beiden Bäume (der Weg wird aus den Quadern gerechnet, ohne die Vorhersage zu rufen, und die Vorhersage prüft jeden Weg gegen: Randpunkt-Falle).
* **Neu-Training:** R1 = 10 % mehr Daten, eine frische Ziehung; R2 bis R11 = gleiche Menge, zehn frische Ziehungen. Ein Weg gilt nach dem Neu-Training nicht mehr, wenn das neue Modell am Zielpunkt Logit > −10⁻⁶ hat.
* **Wahrheit je Funktional (Eingriffs-Lesart, eine Annahme dieser Demo):** der Weg gilt in der Wahrheit nicht, wenn die wahre Wahrscheinlichkeit der Klasse 1 am geänderten Punkt mindestens 0,5 ist (genau dann, wenn die bekannte Funktion μ dort mindestens ln 4 ist).
* **Zähleinheiten:** Anteile ungültiger Wege sind Anteile der **Instanzen mit Weg**; „kein Weg“ ist ein Anteil aller Instanzen. Jede Aussage steht als Spanne über fünf Gruppen zu je 12 Zufalls-Seeds und als Anteil über alle Instanzen (bis zu 1 500).
* **Eingefroren** (JSON mit Quelle und Stand je Feld, `kfk_fragilitaet.json`, erzeugt von `tools/kfk_make_fragilitaet.py`, Rechenzeit der Messung 427 s mit vier Prozessen, nur Orientierung; `--check` rechnet einen Zufalls-Seed neu und vergleicht als Band). **Live** rechnet die App nur den einen Weg (ein Training von n = 1 000 und die Aufzählung; ein bis wenige Sekunden, Spinner).
* **Rangmaß:** ein Weg hat keinen Rang; ein Rangmaß ist hier nicht anwendbar und nicht gemessen. Betragsmaße sind der Anteil ungültiger Wege, die Kosten (Median), der Anteil ohne Weg und die Zahl der Wege.

## Befunde (gemessen, keine Behauptungen)

Politik A (alles veränderbar), Klasse „länger als 4 Tage“, n = 1 000, 60 Zufalls-Seeds in fünf Gruppen zu je 12, bis zu 300 Instanzen je Gruppe, 1 500 insgesamt. Spannen über die Gruppen, in Klammern der Anteil über alle Instanzen mit Weg.

| Modellklasse | Genauigkeit Test | Kosten Median Abstand 0 / 1,0 | ungültig nach R1, Abstand 0 | Abstand 0,5 | Abstand 1,0 | Mittel R2 bis R11, Abstand 0 | in der Wahrheit Abstand 0 / 1,0 | Weg ändert das Gewicht |
|---|---|---|---|---|---|---|---|---|
| Logistische Regression | 79,6 % | 1,00 / 1,20 | 15 bis 22 % (18,2) | 1 bis 6 % (2,7) | 0 % (0,0) | 18 bis 23 % (19,8) | 17 bis 25 / 0 % | 0,3 bis 1,3 % |
| Baum (Tiefe 4) | 74,7 % | 1,00 / 1,20 | 35 bis 44 % (39,3) | 8 bis 35 % (20,1) | 0 bis 24 % (10,3) | 30 bis 40 % (33,9) | 29 bis 37 / 3 bis 10 % | 0,0 % |
| Boosting | 78,6 % | 1,00 / 1,20 | 31 bis 35 % (33,4) | 18 bis 24 % (20,2) | 8 bis 13 % (10,7) | 31 bis 35 % (33,4) | 31 bis 34 / 6 bis 11 % | 4,0 bis 6,3 % |
| Kleines Netz | 78,6 % | 1,00 / 1,19 | 26 bis 33 % (29,6) | 15 bis 18 % (16,6) | 8 bis 10 % (8,7) | 29 bis 31 % (30,1) | 26 bis 30 / 2 bis 6 % | 2,7 bis 7,3 % |
| Exakt optimaler Baum | 77,0 % | 1,00 / 1,20 | 15 bis 40 % (26,2) | 14 bis 37 % (24,9) | 11 bis 34 % (22,1) | 15 bis 28 % (23,9) | 30 bis 36 / 21 bis 27 % | 0,0 bis 0,7 % |
| Exakt optimaler Baum, zweite Regel (größter Index) | 77,1 % | 1,00 / 1,20 | 15 bis 42 % (27,2) | 14 bis 37 % (25,3) | 11 bis 34 % (22,1) | 17 bis 29 % (24,0) | 30 bis 37 / 21 bis 26 % | 0,0 bis 0,7 % |

**Was die Tabelle sagt, mit ihren Voraussetzungen.** (1) Der billigste Weg gilt nach dem Neu-Training bei Abstand 0 in allen fünf Klassen für einen erheblichen Teil der Instanzen nicht mehr (Anteil über alle Instanzen mit Weg 18 bis 39 %); die Reihung der Klassen ist grob (Baum und Boosting vorn, Regression hinten, die Spannen überlappen stark). (2) Ein Sicherheitsabstand von 1,0 senkt den Anteil bei der **logistischen Regression** nach R1 in jeder Gruppe auf 0 % (in der zweiten Messung 0 bis 1 %, nicht im Test), bei den anderen vier Klassen aber nicht in allen Gruppen auf höchstens 3 % (Spannen 0 bis 24, 8 bis 13, 8 bis 10, 11 bis 34 %; beim Baum liegt eine Gruppe bei 0 %, über alle Instanzen sind es 10,3 %). Er kostet etwa 20 % mehr (Median 1,20 gegen 1,00; beim Netz 1,19). (3) Beim **exakt optimalen Baum** streut R1 zwischen den Gruppen stärker als der Mittelwert über zehn Ziehungen: R1 gibt je Gruppe 15 bis 40 %, die Mittel der zehn Ziehungen gleicher Menge 15 bis 28 %; über alle Instanzen sind beide ähnlich (26,2 gegen 23,9 %); ein exakt lesbarer Baum macht den Weg nicht wahrer (in der Wahrheit 30 bis 36 % ungültig, mit Abstand 1,0 noch 21 bis 27 %). (4) Die **zweite Gleichstandsregel** ändert die Spannen des exakt optimalen Baums kaum (15 bis 40 gegen 15 bis 42 %), die Funktion außerhalb der Trainingszeilen aber schon (auf dem Zufalls-Seed 1 sind es zwei verschiedene Bäume mit gleichem Fehler). (5) Der billigste Weg ändert bei Boosting in 4,0 bis 6,3 % und beim Netz in 2,7 bis 7,3 % der Instanzen auch das **Gewicht** (allein oder neben anderen Merkmalen), ein Merkmal ohne Wirkung: ein Modellfehler, den der billigste Weg findet. Wirkungslos ist nur dieser Teil des Wegs; die übrigen geänderten Merkmale können wirken.

**Kein Weg (Abstand 0, Anteil aller Instanzen, Spanne über die Gruppen und die fünf Klassen):** Politik A (alles veränderbar) 0 %, Politik B (Vorlauf und Zoll) 30 bis 45 %, Politik C (nur Vorlauf) 41 bis 55 %. Ob es einen Weg gibt, hängt stärker an der Politik als am Modell; die Politik ist eine Annahme des Verfassers.

**Ungültig nach R1 je Politik (alle drei Abstände, Spanne über die Gruppen):** Bei Politik A liegt der Anteil in jeder Gruppe und in allen Klassen (auch bei der zweiten Gleichstandsregel) bei höchstens 44 %. Bei den engeren Politiken liegt der exakt optimale Baum in einzelnen Gruppen über 50 %: bis 65 % (Politik B bei Abstand 0 bis 55 %, bei Abstand 1,0 bis 65 %; Politik C bis 65 %). Der Satz „zwischen 0 und 1/2“ gilt also nur bei Politik A, und die geschlossene Form der App gilt nur bei gleichverteiltem Spielraum.

**Lage des Wegs zur Grenze (Politik A, Abstand 0, Zufalls-Seeds 40000 bis 40005 mit je bis zu 25 Instanzen, 150 Wege je Klasse; `test_claims.py` rechnet sie neu, sie sind nicht Teil der eingefrorenen Messung):** Das Logit nach dem Weg liegt im Median bei −0,44 (Regression), −0,54 (Boosting) und −0,64 (Baum); innerhalb von 0,1 der Entscheidungsgrenze liegen 12, 12 und 0 % der Wege. Der billigste Weg ist der billigste nach den gesetzten Kosten, nicht der nächste an der Grenze; warum das so ist, ist nicht untersucht (naheliegend: Kategorien springen, und das Raster ist diskret). Die Zahlen sind die des Entwicklungsrechners; der Test prüft sie als Band (Median zwischen −0,8 und −0,3, höchstens 20 % innerhalb von 0,1).

**Mehr als ein Weg (Abstand 0, Politik A, Mengen geänderter Merkmale innerhalb 10 % der Mindestkosten):** Regression 14 bis 19 %, Baum 0 bis 3 %, Boosting 20 bis 27 %, Netz 20 bis 23 %, exakt optimaler Baum 5 bis 7 %. **Im Tab „Gleich gute Wege“** (scikit-learn-Baum, n = 3 000, bis zu 60 Instanzen je Zufalls-Seed, 30 Zufalls-Seeds in fünf Gruppen zu je 6, Weg nur auf Rasterpunkten): Schwelle 10 Tage, Tiefe 5, Zählung nach Blättern 61 bis 87 % (75,6 % über alle 324 Instanzen), nach Mengen geänderter Merkmale 50 bis 87 %; Schwelle 4 Tage, Tiefe 5: nach Blättern 2 bis 16 %, nach Mengen 1 bis 12 %; Tiefe 3 bei Schwelle 4 Tage: 0 %. Die Zahl hängt an der Zählweise, an der Schwelle und an der Tiefe; sie ist keine Eigenschaft von Gegenfaktischen. Bei Schwelle 10 Tage gibt es je Zufalls-Seed nur wenige Instanzen mit Weg (191 bis 327 über alle 30 Zufalls-Seeds).

**Zahl → Test.** Jede Zahl der Befundtabelle und der Absätze darunter („Kein Weg“, „Mehr als ein Weg“, Tab „Gleich gute Wege“) und die Spalte „Bau“ der Korrekturtabelle prüft `tests/test_claims.py` über dieselben Auswertungsfunktionen wie die App (`kfk_evaluation.band`, `tab_band`) gegen die eingefrorene Messung. **Nicht** im Test stehen die Spalte „Vormessung“ der Korrekturtabelle, die zweite Messung (Messung dieser README) und die Vormessung mit dem Kreditscore; die Zahlen der Abschnitte „Modell“ und „Methodik“ stehen nur teilweise im Test (Kandidatenzahl und Schnitte in den Tests der Rechenkerne, die Rechenzeit gegen die Messdatei). Die Lage des Wegs zur Grenze und die Zahl der nicht konvergierten Netze rechnet `test_claims.py` live nach, als Band. Orakel: ganzzahliges Programm gleich Aufzählung in 108 von 108 Fällen, Aufzählung gleich unabhängige Python-Schleife über das rohe Raster, Blattquader gleich Aufzählung in 45 von 45 Fällen je Baum, DP gleich Aufzählung aller Bäume (Klassifikation und Regression, Tiefe 2 und 3), geschlossene Form der Ungültigkeit 0,2464 gleich Integration und Simulation.

## Ehrliche Grenzen

Die Tabelle steht wortgleich im Abschnitt „🚧 Wo die Annahmen enden“ der App (ein Test vergleicht beide Fassungen).

| Annahme | Was passiert, wenn sie verletzt ist | Gilt für |
|---|---|---|
| **Die Kosten sind gesetzt** (1 je geänderte Kategorie oder Binärgröße, Vorlauf \|Δ\| geteilt durch die Standardabweichung der Vorlaufzeiten, Gewicht \|Δ\|) | Andere Gewichte sind nicht gemessen; der Median der Kosten von rund 1 hängt an dieser Wahl. | alle Kosten der Demo |
| **Die Liste der veränderbaren Merkmale (Politik) und das Raster sind vom Verfasser gewählt** | Ob es einen Weg gibt, hängt stärker an der Politik als am Modell; mit dem Raster (0,5-Tage-Schritte) ändern sich Gleichstände und Zählungen. Bei Umschlag erlaubt das Raster Vorlauf bis 10 Tage, der Generator zieht nur bis 6 Tage; ob das Wege verschiebt, ist nicht gemessen. | Experiment 2 und die Zahl der Wege |
| **Eingriffs-Lesart der Wahrheit**: die Änderung wird als Eingriff im Strukturmodell des Generators gelesen | Das ist eine Annahme dieser Demo, kein Satz über die Wirklichkeit; die kausale Fassung (Karimi u. a.) ist nicht gebaut. | zweite Anzeige in Experiment 3 |
| **Ein erfundener Generator, n = 1 000 Trainingszeilen, Klasse „länger als 4 Tage“** | Die Anteile gelten für diese Wahl; ein zweites Vehikel (Kreditscore) ist nur in der README als Gegenprobe genannt, nicht in der App. | alle Anteile |
| **Ein Neu-Training ist ein Modell derselben Klasse auf frischen Daten** | Streuung des Modells bei gleichen Daten (anderer Zufalls-Seed von Boosting und Netz) ist nicht getrennt gemessen. | Experiment 3 |
| **Der Weg ist der billigste, nicht der robusteste oder vielfältigste** | Robuste Wege (ROAR) und vielfältige Wege (DiCE) sind nicht gebaut. | Experiment 1 und 3 |
| **Nutzen in Betrieb oder Recht** | In den gelesenen Quellen fand sich kein Beleg, dass kontrafaktische Erklärungen in Betrieb oder Logistik bessere Entscheidungen liefern; die Demo behauptet das nicht und macht keine Rechtsaussage. | die ganze Demo |

Zusätzlich: Das kleine Netz erreicht das Iterationslimit (500) meist ohne Konvergenz: von 12 Trainings (Zufalls-Seeds 40000 bis 40011) taten es 11 (auf dem Entwicklungsrechner; `test_claims.py` zählt neu und verlangt als Band mindestens 9 von 12); in der eingefrorenen Messung ist es nicht gezählt. Es wird trotzdem verwendet, in der Messung wie in der App (die App meldet es). Zweites Vehikel: in einer Vormessung (erfundener Kreditscore, logistische Regression) waren 22 bis 36 % der Wege nach einem Neu-Training ungültig; das ist eine Vormessung, hier nicht neu gerechnet und nicht in der App.

## Verifikation

* **Kern:** Weg gegen Handrechnung in Brüchen (Handbeispiele Teil 1 bis 4: 84 Kandidaten, Kosten 5/2, 11/4, 3, Logit 1/20 und −1/2, 13 von 13 und 0 von 13, Baum 5/4), Orakel gegen einen anderen Rechenweg (siehe oben), Eigenschaften je Zufalls-Seed (engere Politik gibt nie weniger „kein Weg“; größerer Abstand kostet nie weniger), Randfälle (Instanz schon Klasse 0, Abstand jenseits jeder erreichbaren Distanz, leere Politik, Rasterrand, Gleichstand, Modell mit Klasse 1 überall), Gegenprobe jedes Wegs mit der Modellvorhersage.
* **Baum:** DP gegen Aufzählung aller Bäume (Handbeispiel Teil 6: 5 Bäume bis Tiefe 1, 42 bis Tiefe 2, 2 optimale mit Fehler 0), beide Gleichstandsregeln gegen den kleinsten und größten Index unter allen optimalen Bäumen; eingefrorene Werte für die Kopie in einem Folgestück (`tests/test_tree_frozen.py`).
* **Generator:** Kopie-Test der Formel gegen Werte des Originals (`verweilzeit-demo`), Zufall gegen eingefrorene Folgen (SplitMix64), Verteilung im Band mit Standardfehler.
* **Messung:** Aufbau und innere Widersprüche der eingefrorenen Datei, Neu-Rechnen eines Zufalls-Seeds als Band.
* **Presets, Oberfläche, Layout, Footer, Laufzeit-Importe:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Presets vollständig und einzigartig, Schnellstart in der Seitenleiste mit Hervorhebung, Verdikte als reine Funktionen an handgesetzten Werten.
* **Testzahl:** 151 Tests (`pytest --collect-only -q tests`), Laufzeit der Suite rund 3 Minuten ohne Parallelisierung (gemessen 171 s am 2026-11-20, Orientierung, Maschine unter Last; unter Windows mit Python 3.14 stürzte der Interpreter in einzelnen Läufen mit „access violation“ ab, mit `-n 4`, mit `-n 2` und einmal seriell; ein Wiederholungslauf war jeweils grün). **Browser:** die App wurde einmal im Browser-Fenster gestartet (Port 10073): die Seite lädt ohne Fehlermeldung, das Preset „Exakt optimaler Baum“ wurde angeklickt, der Tab „Rechenweg“ gelesen; alle übrigen Presets und Tabs sind nur mit AppTest geprüft, kein Mausklick auf Diagramme.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Experimente 1 bis 3, Tab „Gleich gute Wege“, Expander) |
| `kfk_generator.py`, `kfk_rng.py` | Verweilzeit-Generator (Kopie der Formel aus `verweilzeit-demo`), SplitMix64 |
| `kfk_recourse.py` | Raster, Kosten, Politik, Aufzählung, ganzzahliges Programm, Weg über Blattquader |
| `kfk_tree.py` | exakt optimaler Baum: DP, Aufzählung aller Bäume als Orakel, Gleichstandsregel, Blattquader |
| `kfk_models.py` | die fünf Modellklassen |
| `kfk_fragility.py` | Messprotokoll (Neu-Training, Wahrheit, gleich gute Wege) |
| `kfk_fragilitaet.json` | eingefrorene Messung (Quelle und Stand in `meta`) |
| `tools/kfk_make_fragilitaet.py` | erzeugt die Messung, `--check` prüft sie |
| `kfk_evaluation.py` | Bänder, Live-Weg, Verdikte, geschlossene Form |
| `kfk_constants.py`, `kfk_presets.py`, `kfk_layout.py`, `kfk_texts.py`, `kfk_visualization.py` | Konstanten und Presets, Permalink, fester Kopf, geteilte Texte, Abbildungen |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Laufzeit (ungepinnt), Tests, CI |

## Bewusst nicht umgesetzt / nicht gemessen

🚧 Nicht gemessen oder nicht gebaut: andere Kostengewichte; Rauschen im Modell selbst bei gleichen Daten; vielfältige (DiCE) und robuste (ROAR) Wege; kausale Fassung (Karimi u. a.); Baumensembles als ganzzahliges Programm (Parmentier und Vidal; die Aufzählung genügt für sechs Merkmale); zweites Vehikel in der App; Menschenstudien (nur Lehrseite); Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün, die Neu-Rechnen-Tests sind als Band geschrieben; Mausklick-Prüfung aller Presets und Tabs im Browser (siehe „Testzahl“).

## Quellen (Stufe der Lektüre: A = Zusammenfassung gelesen, sonst „nicht gelesen“)

Wachter, Mittelstadt, Russell 2018 (kleinste Änderung der Welt, die das gewünschte Ergebnis liefert; A). Ustun, Spangher, Liu 2019 (veränderbare Merkmale, ganzzahlige Programme für lineare Klassifikation; A). Rawal, Kamar, Lakkaraju (Modellupdates machen Wege ungültig; A). Upadhyay u. a. 2021, „ROAR“ (robuste Wege; A). Pawelczyk u. a. 2020 (Modell-Vielfachheit; A). Mothilal u. a., DiCE (Vielfalt; A). Parmentier und Vidal 2021 (ganzzahlige Programme für Baumensembles; A, nicht gebaut). Karimi u. a., Survey (A; die kausale Fassung nicht gelesen). Molnar, *Interpretable Machine Learning*, Kapitel zu kontrafaktischen Erklärungen (nur Seitenzusammenfassung). Dies ist **keine neue Methode**: alles ist Lehrbuch oder Forschung; neu ist nur die nachrechenbare Vorführung mit Zähleinheit, fünf Modellklassen und einer Wahrheit (Vorbilder: XAI-Bench, OpenXAI, Hooker). Preprints von 2026 sind unbegutachtet.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und scikit-learn.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
