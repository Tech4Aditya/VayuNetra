from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]

CATEGORIES = [
    "Depression", "Deep Depression", "Cyclonic Storm",
    "Severe Cyclonic Storm", "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm", "Super Cyclonic Storm"
]

RAW_TO_CANONICAL = {
    "D":"Depression","Depression":"Depression",
    "DD":"Deep Depression","Deep Depression":"Deep Depression",
    "CS":"Cyclonic Storm","Cyclonic Storm":"Cyclonic Storm",
    "SCS":"Severe Cyclonic Storm","Severe Cyclonic Storm":"Severe Cyclonic Storm",
    "VSCS":"Very Severe Cyclonic Storm","Very Severe Cyclonic Storm":"Very Severe Cyclonic Storm",
    "ESCS":"Extremely Severe Cyclonic Storm","Extremely Severe Cyclonic Storm":"Extremely Severe Cyclonic Storm",
    "SuCS":"Super Cyclonic Storm","Super Cyclonic Storm":"Super Cyclonic Storm",
}
CAT_TO_ID = {x:i for i,x in enumerate(CATEGORIES)}

STORMS = [
    "FANI","TAUKTAE","AMPHAN","YAAS","ASANI",
    "NISARGA","BULBUL","GULAAB","BIPARJOY","MICHAUNG"
]

FOLDS = [
    ("FOLD_01", ["FANI","NISARGA"]),
    ("FOLD_02", ["AMPHAN","GULAAB"]),
    ("FOLD_03", ["BIPARJOY","BULBUL"]),
    ("FOLD_04", ["ASANI","YAAS"]),
    ("FOLD_05", ["TAUKTAE","MICHAUNG"]),
]

def load_all():
    frames=[]
    for storm in STORMS:
        p=ROOT/"data"/"labels"/f"{storm.lower()}_insat_manifest.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        d=pd.read_csv(p)
        d["storm"]=storm
        d["canonical_category"]=d["category"].map(RAW_TO_CANONICAL)
        if d["canonical_category"].isna().any():
            raise ValueError(f"Unmapped labels in {storm}")
        d["category_id"]=d["canonical_category"].map(CAT_TO_ID)
        frames.append(d)
    return pd.concat(frames,ignore_index=True)

def validation_storms(df,test_storms):
    remaining=[s for s in STORMS if s not in test_storms]
    return sorted(
        remaining,
        key=lambda s:(-len(df[df.storm==s]),s)
    )[:2]

def main():
    df=load_all()
    results=[]
    pooled_true=[]
    pooled_pred=[]

    print("="*72)
    print("INSAT 5-FOLD MAJORITY BASELINE")
    print("="*72)

    for fold,test_storms in FOLDS:
        val_storms=validation_storms(df,test_storms)
        train_storms=[
            s for s in STORMS
            if s not in test_storms and s not in val_storms
        ]

        train=df[df.storm.isin(train_storms)]
        test=df[df.storm.isin(test_storms)]

        counts=train.category_id.value_counts().reindex(
            range(7),fill_value=0
        )
        majority=int(counts.idxmax())

        y=test.category_id.to_numpy()
        pred=np.full(len(test),majority,dtype=np.int64)

        acc=accuracy_score(y,pred)
        f1=f1_score(y,pred,average="macro",zero_division=0)
        mae=float(np.mean(np.abs(y-pred)))
        cm=confusion_matrix(y,pred,labels=list(range(7)))

        pooled_true.extend(y)
        pooled_pred.extend(pred)

        print("\n"+"="*72)
        print(fold)
        print("="*72)
        print("TRAIN:",train_storms,len(train))
        print("VAL  :",val_storms,len(df[df.storm.isin(val_storms)]))
        print("TEST :",test_storms,len(test))
        print("Majority:",CATEGORIES[majority])
        print(f"Accuracy     : {acc:.4f}")
        print(f"Macro F1     : {f1:.4f}")
        print(f"Category MAE : {mae:.4f}")

        results.append({
            "fold":fold,
            "train_storms":train_storms,
            "val_storms":val_storms,
            "test_storms":test_storms,
            "train_samples":int(len(train)),
            "test_samples":int(len(test)),
            "majority_class":CATEGORIES[majority],
            "majority_class_id":majority,
            "metrics":{
                "accuracy":float(acc),
                "macro_f1":float(f1),
                "category_mae":mae
            },
            "training_category_counts":{
                CATEGORIES[i]:int(counts.iloc[i]) for i in range(7)
            },
            "confusion_matrix":cm.tolist()
        })

    vals={m:np.array([r["metrics"][m] for r in results])
          for m in ["accuracy","macro_f1","category_mae"]}

    yt=np.asarray(pooled_true)
    yp=np.asarray(pooled_pred)
    pooled_cm=confusion_matrix(yt,yp,labels=list(range(7)))

    report={
        "model":"Storm-level majority baseline",
        "dataset":"NIO 10-storm INSAT",
        "total_samples":int(len(df)),
        "folds":results,
        "aggregate":{
            m:{
                "mean":float(v.mean()),
                "std":float(v.std()),
                "values":[float(x) for x in v]
            } for m,v in vals.items()
        },
        "pooled":{
            "accuracy":float(accuracy_score(yt,yp)),
            "macro_f1":float(f1_score(yt,yp,average="macro",zero_division=0)),
            "category_mae":float(np.mean(np.abs(yt-yp))),
            "confusion_matrix":pooled_cm.tolist()
        }
    }

    out=ROOT/"data"/"processed"/"insat_majority_baseline_cv.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")

    print("\n"+"="*72)
    print("5-FOLD MAJORITY BASELINE SUMMARY")
    print("="*72)
    for m,v in vals.items():
        print(f"{m:18s}: {v.mean():.4f} +/- {v.std():.4f}")
    print(f"{'pooled_accuracy':18s}: {accuracy_score(yt,yp):.4f}")
    print(f"{'pooled_macro_f1':18s}: {f1_score(yt,yp,average='macro',zero_division=0):.4f}")
    print(f"{'pooled_category_mae':18s}: {np.mean(np.abs(yt-yp)):.4f}")
    print("\nReport:",out)

if __name__=="__main__":
    main()
