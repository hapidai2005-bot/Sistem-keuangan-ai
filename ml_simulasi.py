"""Demo Random Forest: riwayat sintetis, tidak ditulis ke database operasional."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURES=['nominal','due_month','prior_late','prior_count']

def make_demo_history(current, seed=2026):
    """Bangun 6 periode fiktif per siswa dari profil siswa dan nominal pada dataset asli.
    Label ditentukan dari tanggal bayar simulasi, bukan dari tagihan asli.
    """
    rng=np.random.default_rng(seed)
    students=current.sort_values('id').drop_duplicates('siswa_id')
    rows=[]
    for _,s in students.iterrows():
        tendency=float(rng.uniform(.08,.76))
        prior_late=0
        for idx,month in enumerate(range(4,10)):
            due=pd.Timestamp(2026,month,10)
            late=bool(rng.random()<tendency)
            offset=int(rng.integers(1,19)) if late else -int(rng.integers(0,8))
            paid=due+pd.Timedelta(days=offset)
            rows.append({'siswa_id':s.siswa_id,'nis':s.nis,'nama':s.nama,'kelas':s.kelas,
                         'nominal':float(s.nominal),'due':due,'tanggal_bayar':paid,
                         'due_month':month,'prior_late':prior_late/idx if idx else 0.,
                         'prior_count':idx,'late':int(paid>due),'sumber':'SIMULASI'})
            prior_late+=int(late)
    return pd.DataFrame(rows)

def train_demo(history):
    months=sorted(history.due_month.unique())
    train=history[history.due_month.isin(months[:-2])]
    test=history[history.due_month.isin(months[-2:])]
    if train.late.nunique()<2 or test.late.nunique()<2:
        raise ValueError('Variasi label simulasi tidak mencukupi.')
    model=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,max_depth=7,
                                 class_weight='balanced',random_state=42,n_jobs=-1)
    model.fit(train[FEATURES],train.late)
    pred=model.predict(test[FEATURES])
    metrics={'akurasi':accuracy_score(test.late,pred),'presisi':precision_score(test.late,pred,zero_division=0),
             'recall':recall_score(test.late,pred,zero_division=0),'f1':f1_score(test.late,pred,zero_division=0),
             'confusion':confusion_matrix(test.late,pred,labels=[0,1])}
    # Model final hanya menggunakan riwayat simulasi yang sudah memiliki label.
    model.fit(history[FEATURES],history.late)
    return model,metrics

def score_current(model, history, current):
    stats=history.groupby('siswa_id').agg(prior_late=('late','mean'),prior_count=('late','size')).reset_index()
    result=current.merge(stats,on='siswa_id',how='left')
    result['prior_late']=result.prior_late.fillna(0)
    result['prior_count']=result.prior_count.fillna(0)
    result['due_month']=pd.to_datetime(result.jatuh_tempo).dt.month
    result['skor_simulasi']=(model.predict_proba(result[FEATURES])[:,list(model.classes_).index(1)]*100).round(1)
    result['kategori_simulasi']=pd.cut(result.skor_simulasi,bins=[-1,35,70,101],labels=['Rendah','Sedang','Tinggi'])
    return result
