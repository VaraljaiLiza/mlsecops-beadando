# MLSecOps beadandó

Használt autók árkategóriájának előrejelzése a CarDekho adatai alapján.
A projekt bemutatja az adatverziózást, a modelltanítást, a kísérletkövetést,
az API-alapú előrejelzést és a Dockerrel történő indítást.

## Adatok és modell

- Forrás: https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data
- A nyers adatállományból 3000 soros véletlen mintát készítünk.
- A mintavétel véletlen magja: 42.
- Feladat: bináris klasszifikáció.
- Célváltozó: `price_category`.
- Az árkategóriák képzésének küszöbe: 600000, a forrásadat árának egységében.
- Teszthalmaz aránya: 20%.
- Modell: Random Forest, 100 fával, legfeljebb 10-es mélységgel.

Numerikus jellemzők: vehicle_age, km_driven, mileage, engine, max_power, seats.

Kategóriás jellemzők: brand, model, seller_type, fuel_type, transmission_type.

A beállításokat a `params.yaml` tartalmazza.

## Felhasznált eszközök

- Git és GitHub: kódverziózás.
- DVC: a nyers adatok és a feldolgozott minta verziókövetése.
- MLflow: kísérletek, metrikák és modellverziók nyilvántartása.
- FastAPI: a modell elérése HTTP API-n keresztül.
- Evidently és Streamlit: monitoring és dashboard.
- Docker Compose: a szolgáltatások összehangolt indítása.
- unittest, coverage és GitHub Actions: automatikus tesztelés.

A modellaktiválás összehasonlítási mérőszáma a `balanced_accuracy`.
A konfigurált minimális javulás 0.0; az aktiválási szabályokat
a `src/promotion.py` tartalmazza.

## Előfeltételek

- Git
- Python 3.13, venv és pip támogatással
- Docker Engine és Docker Compose plugin
- Internetkapcsolat az adatok, csomagok és Docker-alapképek letöltéséhez

Az alábbi parancsok Linux/Bash környezetre készültek.

## Indítás friss gépen

```bash
git clone https://github.com/VaraljaiLiza/mlsecops-beadando.git
cd mlsecops-beadando

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install dvc==3.67.1

dvc pull
docker compose up -d --build
```

A DVC a nyilvános `mlsecops-data` GitHub-repositoryból tölti le az adatokat.
A letöltéshez nincs szükség hozzáférési tokenre.

Az első Docker-build a függőségek telepítése miatt több percig tarthat.
A Docker a konténeren belüli Python-függőségeket automatikusan telepíti.

Az indítás sorrendje:

1. Elindul az MLflow.
2. A bootstrap szolgáltatás elvégzi az előfeldolgozást és a tanítást.
3. A bootstrap sikeres befejezése után elindul az API.
4. Az API egészséges állapota után elindul a dashboard.

## Szolgáltatások elérése

A szolgáltatások a Dockert futtató gépen érhetők el:

| Szolgáltatás | Cím |
| --- | --- |
| MLflow | http://127.0.0.1:5000 |
| API dokumentáció és kipróbálás | http://127.0.0.1:8000/docs |
| API állapot | http://127.0.0.1:8000/health |
| Dashboard | http://127.0.0.1:8501 |

Virtuális gép használatakor ezek a címek a virtuális gépen értendők.

Az API interaktív dokumentációjában megtekinthetők a végpontok,
a szükséges bemeneti mezők és a válaszok formátumai.

## Állapot és naplók

```bash
docker compose ps -a
docker compose logs --tail 100 bootstrap
docker compose logs --tail 100 api
```

A bootstrap egyszer fut le; sikeres befejezésekor az állapota `Exited (0)`.
Az API és az MLflow szolgáltatásnál `healthy` állapot várható.

## Tesztelés

A projekt gyökérmappájában, aktivált virtuális környezetben:

```bash
python -m pip install -r requirements.txt
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
```

Az elvárt minimális lefedettség 95%.
A mérés a `src`, `scripts` és `tests` könyvtárakat tartalmazza.

A GitHub Actions a main ágra irányuló Pull Requesteknél és a main ág
frissítésekor automatikusan lefuttatja a teszteket és a lefedettség-ellenőrzést.

## Leállítás és tárolás

```bash
docker compose down
```

A parancs eltávolítja a konténereket, de megtartja a névvel ellátott köteteket.
Ezek tárolják az MLflow-adatokat, a datasetet, az előrejelzési naplókat
és a monitoringjelentéseket.

A `docker compose down -v` ezeket a köteteket és a bennük lévő adatokat is törli.

## Adatverziók feltöltése

A letöltéshez használt GitHub HTTPS-cím csak olvasásra szolgál.
Az új adatverziók feltöltésének menete:

1. Az adatok módosítása és követése `dvc add` segítségével.
2. `dvc push` egy helyileg beállított, írható adattárolóba.
3. Az adattároló repository változtatásainak feltöltése Git segítségével.
4. A frissített `.dvc` hivatkozások mentése a beadandó repositoryjába.

Adattároló: https://github.com/VaraljaiLiza/mlsecops-data
