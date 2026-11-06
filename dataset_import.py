"""Import Excel monitoring pembayaran ke SQLite; validasi dahulu, commit atomik."""
import io
import sqlite3
from datetime import date, datetime
import pandas as pd

SHEETS = {
 'Data_Siswa': ['ID_Siswa','NIS','Nama_Siswa','Kelas'],
 'Data_Tagihan': ['ID_Tagihan','ID_Siswa','Periode','Jenis_Tagihan','Total_Tagihan','Jatuh_Tempo'],
 'Data_Pembayaran': ['ID_Pembayaran','ID_Tagihan','ID_Siswa','Tanggal_Bayar','Jumlah_Bayar','Metode_Bayar'],
}

def read_excel(content):
    book = pd.ExcelFile(io.BytesIO(content), engine='openpyxl')
    missing = [s for s in SHEETS if s not in book.sheet_names]
    if missing: raise ValueError('Sheet tidak ditemukan: '+', '.join(missing))
    data={s:pd.read_excel(book,sheet_name=s,dtype={'NIS':str,'ID_Siswa':str,'ID_Tagihan':str,'ID_Pembayaran':str}) for s in SHEETS}
    for name, required in SHEETS.items():
        absent=[c for c in required if c not in data[name].columns]
        if absent: raise ValueError(f'Sheet {name}: kolom wajib tidak ditemukan: {", ".join(absent)}')
        data[name]=data[name].dropna(how='all').copy()
    return data

def val(x):
    if pd.isna(x): return ''
    return str(x).strip()

def iso(x):
    if pd.isna(x) or str(x).strip()=='': raise ValueError('Tanggal wajib diisi')
    t=pd.to_datetime(x,errors='coerce')
    if pd.isna(t): raise ValueError(f'Tanggal tidak valid: {x}')
    return t.date().isoformat()

def money(x):
    n=pd.to_numeric(x,errors='coerce')
    if pd.isna(n) or n<=0 or n!=int(n): raise ValueError(f'Nominal tidak valid: {x}')
    return int(n)

def validate(data):
    students,bills,payments=(data[k] for k in SHEETS)
    for name,df,col in [('Data_Siswa',students,'ID_Siswa'),('Data_Siswa',students,'NIS'),('Data_Tagihan',bills,'ID_Tagihan'),('Data_Pembayaran',payments,'ID_Pembayaran')]:
        ids=df[col].map(val)
        if ids.eq('').any() or ids.duplicated().any(): raise ValueError(f'{name}: {col} kosong atau duplikat')
    sid=set(students.ID_Siswa.map(val)); tid=set(bills.ID_Tagihan.map(val))
    if not set(bills.ID_Siswa.map(val)).issubset(sid): raise ValueError('Tagihan mengacu ke ID_Siswa yang tidak ada')
    if not set(payments.ID_Tagihan.map(val)).issubset(tid): raise ValueError('Pembayaran mengacu ke ID_Tagihan yang tidak ada')
    mapping=dict(zip(bills.ID_Tagihan.map(val),bills.ID_Siswa.map(val)))
    totals={}
    for _,r in bills.iterrows():
        if not all(val(r[c]) for c in ['ID_Tagihan','ID_Siswa','Periode','Jenis_Tagihan']): raise ValueError('Data tagihan memiliki kolom wajib kosong')
        iso(r['Jatuh_Tempo']); totals[val(r['ID_Tagihan'])]=money(r['Total_Tagihan'])
    paid={k:0 for k in totals}
    for _,r in payments.iterrows():
        k=val(r['ID_Tagihan']); s=val(r['ID_Siswa'])
        if s and mapping[k]!=s: raise ValueError(f'ID siswa pembayaran tidak sesuai tagihan {k}')
        iso(r['Tanggal_Bayar']); paid[k]+=money(r['Jumlah_Bayar'])
    over=[k for k in totals if paid[k]>totals[k]]
    if over: raise ValueError('Pembayaran melebihi nilai tagihan: '+', '.join(over[:8]))
    for _,r in students.iterrows():
        if not all(val(r[c]) for c in ['NIS','Nama_Siswa','Kelas']): raise ValueError('Data siswa memiliki NIS/nama/kelas kosong')
    return {'siswa':len(students),'tagihan':len(bills),'pembayaran':len(payments),'total_tagihan':sum(totals.values()),'total_bayar':sum(paid.values())}

def import_data(db_path,data,replace=False):
    summary=validate(data)
    c=sqlite3.connect(db_path)
    try:
        c.execute('PRAGMA foreign_keys=ON')
        c.execute('BEGIN IMMEDIATE')
        if replace:
            c.execute('DELETE FROM pembayaran');c.execute('DELETE FROM tagihan');c.execute('DELETE FROM siswa')
        elif any(c.execute(f'SELECT 1 FROM {table} LIMIT 1').fetchone() for table in ('siswa','tagihan','pembayaran')):
            raise ValueError('Database sudah berisi data. Untuk mengimpor dataset penuh, pilih opsi Ganti semua data setelah membuat cadangan.')
        student_ids={}
        for _,r in data['Data_Siswa'].iterrows():
            active=val(r.get('Status_Siswa','Aktif')).lower() not in ('nonaktif','tidak aktif','lulus')
            cur=c.execute('INSERT INTO siswa(nis,nama,kelas,wali,kontak,aktif) VALUES (?,?,?,?,?,?)',
                (val(r['NIS']),val(r['Nama_Siswa']),val(r['Kelas']),'','',int(active)))
            student_ids[val(r['ID_Siswa'])]=cur.lastrowid
        bill_ids={}
        for _,r in data['Data_Tagihan'].iterrows():
            cur=c.execute('INSERT INTO tagihan(siswa_id,jenis,periode,nominal,jatuh_tempo) VALUES (?,?,?,?,?)',
                (student_ids[val(r['ID_Siswa'])],val(r['Jenis_Tagihan']),val(r['Periode']),money(r['Total_Tagihan']),iso(r['Jatuh_Tempo'])))
            bill_ids[val(r['ID_Tagihan'])]=cur.lastrowid
        for _,r in data['Data_Pembayaran'].iterrows():
            c.execute('INSERT INTO pembayaran(tagihan_id,tanggal,nominal,metode,catatan) VALUES (?,?,?,?,?)',
                (bill_ids[val(r['ID_Tagihan'])],iso(r['Tanggal_Bayar']),money(r['Jumlah_Bayar']),val(r['Metode_Bayar']) or 'Lainnya',val(r.get('Keterangan',''))))
        c.commit()
        return summary
    except Exception:
        c.rollback();raise
    finally: c.close()
