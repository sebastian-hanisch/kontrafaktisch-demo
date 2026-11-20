"""Texte der App, die auch in der README stehen (ein Test vergleicht beide Fassungen)."""

LIMITS_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Gilt für |
|---|---|---|
| **Die Kosten sind gesetzt** (1 je geänderte Kategorie oder Binärgröße, Vorlauf \\|Δ\\| geteilt durch die Standardabweichung der Vorlaufzeiten, Gewicht \\|Δ\\|) | Andere Gewichte sind nicht gemessen; der Median der Kosten von rund 1 hängt an dieser Wahl. | alle Kosten der Demo |
| **Die Liste der veränderbaren Merkmale (Politik) und das Raster sind vom Verfasser gewählt** | Ob es einen Weg gibt, hängt stärker an der Politik als am Modell; mit dem Raster (0,5-Tage-Schritte) ändern sich Gleichstände und Zählungen. Bei Umschlag erlaubt das Raster Vorlauf bis 10 Tage, der Generator zieht nur bis 6 Tage; ob das Wege verschiebt, ist nicht gemessen. | Experiment 2 und die Zahl der Wege |
| **Eingriffs-Lesart der Wahrheit**: die Änderung wird als Eingriff im Strukturmodell des Generators gelesen | Das ist eine Annahme dieser Demo, kein Satz über die Wirklichkeit; die kausale Fassung (Karimi u. a.) ist nicht gebaut. | zweite Anzeige in Experiment 3 |
| **Ein erfundener Generator, n = 1 000 Trainingszeilen, Klasse „länger als 4 Tage“** | Die Anteile gelten für diese Wahl; ein zweites Vehikel (Kreditscore) ist nur in der README als Gegenprobe genannt, nicht in der App. | alle Anteile |
| **Ein Neu-Training ist ein Modell derselben Klasse auf frischen Daten** | Streuung des Modells bei gleichen Daten (anderer Zufalls-Seed von Boosting und Netz) ist nicht getrennt gemessen. | Experiment 3 |
| **Der Weg ist der billigste, nicht der robusteste oder vielfältigste** | Robuste Wege (ROAR) und vielfältige Wege (DiCE) sind nicht gebaut. | Experiment 1 und 3 |
| **Nutzen in Betrieb oder Recht** | In den gelesenen Quellen fand sich kein Beleg, dass kontrafaktische Erklärungen in Betrieb oder Logistik bessere Entscheidungen liefern; die Demo behauptet das nicht und macht keine Rechtsaussage. | die ganze Demo |
"""

DATA_NOTE = (
    "Die Container sind **ausgedacht**: Art, Kunde, Zoll, Wochenende und Vorlauf (die Zeitspanne in Tagen bis zur Schiffsabfahrt, nur bei Export und Umschlag) bestimmen die Verweilzeit nach einer bekannten Formel; das Merkmal Gewicht ist reines Rauschen. "
    "Es gibt keine reale Datenquelle."
)
