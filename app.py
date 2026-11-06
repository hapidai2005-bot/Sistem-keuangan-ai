import sqlite3
from pathlib import Path
from datetime import date, timedelta, datetime
import io
import html
from dataset_import import read_excel, validate, import_data
from ml_simulasi import make_demo_history, train_demo, score_current
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title='Anugerah Insani | Pembayaran', page_icon='🏫', layout='wide')
DB = Path(__file__).with_name('sekolah.db')

@st.cache_resource
def init_db():
    with sqlite3.connect(DB) as c:
        c.execute('PRAGMA foreign_keys=ON')
        c.executescript('''
        CREATE TABLE IF NOT EXISTS siswa(id INTEGER PRIMARY KEY AUTOINCREMENT, nis TEXT UNIQUE NOT NULL, nama TEXT NOT NULL, kelas TEXT NOT NULL, wali TEXT DEFAULT '', kontak TEXT DEFAULT '', aktif INTEGER DEFAULT 1);
        CREATE TABLE IF NOT EXISTS tagihan(id INTEGER PRIMARY KEY AUTOINCREMENT, siswa_id INTEGER NOT NULL REFERENCES siswa(id), jenis TEXT NOT NULL, periode TEXT NOT NULL, nominal INTEGER NOT NULL CHECK(nominal>0), jatuh_tempo TEXT NOT NULL, UNIQUE(siswa_id,jenis,periode));
        CREATE TABLE IF NOT EXISTS pembayaran(id INTEGER PRIMARY KEY AUTOINCREMENT, tagihan_id INTEGER NOT NULL REFERENCES tagihan(id), tanggal TEXT NOT NULL, nominal INTEGER NOT NULL CHECK(nominal>0), metode TEXT NOT NULL, catatan TEXT DEFAULT '');
        ''')
    return True
init_db()

def conn():
    c=sqlite3.connect(DB)
    c.execute('PRAGMA foreign_keys=ON')
    return c

def read(sql,params=()):
    with conn() as c:
        return pd.read_sql_query(sql,c,params=params)

def execute(sql,params=()):
    with conn() as c:
        c.execute(sql,params)
        c.commit()

def bills():
    df=read('''SELECT t.id,t.siswa_id,s.nis,s.nama,s.kelas,t.jenis,t.periode,t.nominal,t.jatuh_tempo,
       COALESCE((SELECT SUM(p.nominal) FROM pembayaran p WHERE p.tagihan_id=t.id),0) AS dibayar,
       (SELECT MAX(p.tanggal) FROM pembayaran p WHERE p.tagihan_id=t.id) AS terakhir_bayar
       FROM tagihan t JOIN siswa s ON s.id=t.siswa_id ORDER BY t.jatuh_tempo DESC,t.id DESC''')
    if not df.empty:
        df['sisa']=df.nominal-df.dibayar
        df['status']=np.where(df.sisa==0,'Lunas',np.where(pd.to_datetime(df.jatuh_tempo).dt.date<date.today(),'Tunggakan','Belum lunas'))
    return df

def rupiah(x): return 'Rp {:,.0f}'.format(float(x)).replace(',','.')

def download_csv(df,label,filename):
    st.download_button(label,df.to_csv(index=False).encode('utf-8-sig'),filename,mime='text/csv')

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'Plus Jakarta Sans',sans-serif}
.stApp{background:linear-gradient(155deg,#f4f8ff 0%,#f8fafc 48%,#edf5ff 100%);color:#12213a}
.block-container{padding-top:1.5rem;padding-bottom:3rem;max-width:1500px}
h1,h2,h3{letter-spacing:-.035em;color:#142c50!important;font-weight:800!important}
h1{font-size:2rem!important}
[data-testid="stSidebar"]{background:linear-gradient(175deg,#10284a 0%,#173b6d 58%,#10284a 100%)}
[data-testid="stSidebar"] *{color:#eef5ff!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:12px;padding:7px 9px;transition:background .2s}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#ffffff1b}
[data-testid="stSidebar"] [data-testid="stAlert"]{background:#ffffff14;border:1px solid #ffffff2c}
[data-testid="stMetric"]{background:#fff;padding:18px 19px;border-radius:17px;border:1px solid #e4ecf7;box-shadow:0 9px 28px #193f7810;transition:transform .2s,box-shadow .2s}
[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 14px 30px #193f7820}
[data-testid="stMetricLabel"]{color:#6b7d97;font-weight:600}
[data-testid="stMetricValue"]{color:#173a70;font-weight:800;letter-spacing:-.04em}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px}
[data-testid="stForm"],div[data-testid="stExpander"]{background:white;border:1px solid #e4ecf7;border-radius:16px;padding:12px}
.stButton>button[kind="primary"],.stDownloadButton>button{background:linear-gradient(90deg,#2364d7,#159bd3);color:white;border:0;border-radius:11px;font-weight:700}
.stButton>button{border-radius:11px;transition:transform .15s}
.stButton>button:hover{transform:translateY(-2px)}
[data-testid="stDataFrame"]{border-radius:13px;overflow:hidden;border:1px solid #e4ecf7}
.hero{background:linear-gradient(110deg,#14356a,#2564b8 65%,#128ca9);border-radius:21px;padding:26px 30px;color:white;box-shadow:0 15px 35px #1a4e9829;margin:0 0 20px}
.hero h2{color:white!important;font-size:1.65rem;margin:0 0 6px}.hero p{color:#dceaff;margin:0}
.eyebrow{font-size:.72rem;font-weight:800;letter-spacing:.16em;color:#b9d9ff;margin-bottom:8px}
.brand{font-size:1.3rem;font-weight:800;color:white;line-height:1.3}.brand-sub{font-size:.76rem;color:#c9d8ef;margin-bottom:16px}
.section-caption{color:#6b7d97;font-size:.88rem}
@media(max-width:700px){.block-container{padding:1rem}.hero{padding:21px}.hero h2{font-size:1.35rem}}
</style>""",unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="brand">🏫 ANUGERAH INSANI</div><div class="brand-sub">Smart School Finance • AI Analytics</div>',unsafe_allow_html=True)
    st.caption('MENU UTAMA')
    menu=st.radio('Navigasi',['Dashboard','Data Siswa','Tagihan','Pembayaran','Analisis & Laporan','Prediksi Machine Learning','Upload Dataset Excel','Data Simulasi'],label_visibility='collapsed')
    st.divider()
    st.caption('🔒 Mode prototipe lokal • Belum ada autentikasi')
    st.info('Gunakan data fiktif untuk demo. Amankan akses sebelum menggunakan data pribadi siswa.')

def hero(title,subtitle,eyebrow='SCHOOL FINANCE PLATFORM'):
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div>',unsafe_allow_html=True)

def style_chart(fig):
    fig.update_layout(template='plotly_white',font=dict(family='Arial',color='#263e5c'),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=18,r=18,t=55,b=15),legend_title_text='',
        colorway=['#2463d4','#11a9a4','#f5a642','#e36e83'])
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#eaf0f8')
    return fig

if menu=='Dashboard':
    hero('Dashboard Keuangan','Pantau pembayaran, tagihan, dan tunggakan sekolah secara real-time.')
    df=bills(); n=read('SELECT COUNT(*) n FROM siswa WHERE aktif=1').iloc[0]['n']
    total=int(df.nominal.sum()) if not df.empty else 0
    paid=int(df.dibayar.sum()) if not df.empty else 0
    overdue=int(df.loc[df.status=='Tunggakan','sisa'].sum()) if not df.empty else 0
    a,b,c,d=st.columns(4)
    a.metric('Siswa aktif',n);b.metric('Total tagihan',rupiah(total));c.metric('Telah dibayar',rupiah(paid));d.metric('Tunggakan jatuh tempo',rupiah(overdue))
    if df.empty: st.info('Belum ada tagihan. Tambahkan siswa dan tagihan atau buat data simulasi.')
    else:
        f1,f2=st.columns([2,1])
        kelas_filter=f1.multiselect('Filter kelas',sorted(df.kelas.unique()),default=sorted(df.kelas.unique()),key='dash_kelas')
        jenis_filter=f2.selectbox('Jenis tagihan',['Semua']+sorted(df.jenis.unique().tolist()),key='dash_jenis')
        df=df[df.kelas.isin(kelas_filter)].copy()
        if jenis_filter!='Semua':df=df[df.jenis==jenis_filter].copy()
        if df.empty:
            st.info('Tidak ada data untuk kombinasi filter ini.')
            st.stop()
        st.caption('Grafik dan tabel berikut mengikuti filter; kartu ringkasan di atas menampilkan keseluruhan data.')
        x,y=st.columns(2)
        with x:
            st.plotly_chart(style_chart(px.pie(df,names='status',title='Status tagihan')),use_container_width=True)
        with y:
            z=df.groupby('kelas',as_index=False)[['nominal','dibayar','sisa']].sum()
            st.plotly_chart(style_chart(px.bar(z,x='kelas',y=['dibayar','sisa'],barmode='group',title='Pembayaran dan sisa per kelas')),use_container_width=True)
        st.subheader('Tagihan terbaru')
        st.dataframe(df[['nis','nama','kelas','jenis','periode','nominal','dibayar','sisa','jatuh_tempo','status']].head(20),use_container_width=True,hide_index=True)

elif menu=='Data Siswa':
    hero('Manajemen Data Siswa','Kelola identitas dan status siswa dalam satu tempat.')
    with st.form('add_siswa',clear_on_submit=True):
        a,b,c=st.columns(3)
        nis=a.text_input('NIS *');nama=b.text_input('Nama siswa *');kelas=c.text_input('Kelas *',placeholder='Contoh: 5A')
        wali=st.text_input('Nama wali');kontak=st.text_input('Kontak wali (opsional)')
        if st.form_submit_button('Simpan siswa'):
            if not all([nis.strip(),nama.strip(),kelas.strip()]):st.error('NIS, nama, dan kelas wajib diisi.')
            else:
                try: execute('INSERT INTO siswa(nis,nama,kelas,wali,kontak) VALUES(?,?,?,?,?)',(nis.strip(),nama.strip(),kelas.strip(),wali.strip(),kontak.strip()));st.success('Siswa tersimpan.')
                except sqlite3.IntegrityError:st.error('NIS sudah digunakan.')
    df=read('SELECT * FROM siswa ORDER BY nama')
    st.dataframe(df,hide_index=True,use_container_width=True)
    if not df.empty:
        opts={f"{r.nis} — {r.nama} ({r.kelas})":int(r.id) for r in df.itertuples()}
        with st.expander('Ubah atau nonaktifkan siswa'):
            selected=st.selectbox('Pilih siswa',list(opts));sid=opts[selected]
            r=df[df.id==sid].iloc[0]
            with st.form('edit_siswa'):
                nama2=st.text_input('Nama',value=r['nama']);kelas2=st.text_input('Kelas',value=r['kelas']);wali2=st.text_input('Wali',value=r['wali']);kontak2=st.text_input('Kontak',value=r['kontak']);aktif=st.checkbox('Aktif',value=bool(r['aktif']))
                if st.form_submit_button('Simpan perubahan'):
                    if nama2.strip() and kelas2.strip():execute('UPDATE siswa SET nama=?,kelas=?,wali=?,kontak=?,aktif=? WHERE id=?',(nama2.strip(),kelas2.strip(),wali2.strip(),kontak2.strip(),int(aktif),sid));st.success('Data diperbarui.');st.rerun()
                    else:st.error('Nama dan kelas wajib diisi.')
        download_csv(df,'Unduh data siswa CSV','siswa.csv')

elif menu=='Tagihan':
    hero('Kelola Tagihan','Buat dan pantau tagihan sekolah secara terstruktur.')
    siswa=read('SELECT id,nis,nama,kelas FROM siswa WHERE aktif=1 ORDER BY nama')
    if siswa.empty:st.warning('Tambahkan siswa aktif terlebih dahulu.')
    else:
        opts={f'{r.nis} — {r.nama} ({r.kelas})':int(r.id) for r in siswa.itertuples()}
        with st.form('add_tagihan',clear_on_submit=True):
            selected=st.selectbox('Siswa',list(opts));a,b,c=st.columns(3)
            jenis=a.selectbox('Jenis tagihan',['SPP','Uang Pangkal','Kegiatan','Seragam','Ujian','Lainnya'])
            periode=b.text_input('Periode',value=date.today().strftime('%Y-%m'),help='Contoh: 2026-11 atau 2026-Ganjil')
            nominal=c.number_input('Nominal (Rp)',min_value=1000,step=10000,value=250000)
            due=st.date_input('Jatuh tempo',value=date.today()+timedelta(days=10))
            if st.form_submit_button('Buat tagihan'):
                if not periode.strip():st.error('Periode wajib diisi.')
                else:
                    try:execute('INSERT INTO tagihan(siswa_id,jenis,periode,nominal,jatuh_tempo) VALUES(?,?,?,?,?)',(opts[selected],jenis,periode.strip(),int(nominal),due.isoformat()));st.success('Tagihan dibuat.')
                    except sqlite3.IntegrityError:st.error('Tagihan dengan jenis dan periode ini sudah ada untuk siswa tersebut.')
    df=bills()
    if not df.empty:
        st.dataframe(df[['id','nis','nama','kelas','jenis','periode','nominal','dibayar','sisa','jatuh_tempo','status']],hide_index=True,use_container_width=True)
        download_csv(df,'Unduh tagihan CSV','tagihan.csv')

elif menu=='Pembayaran':
    hero('Transaksi Pembayaran','Catat cicilan, pembayaran penuh, dan bukti transaksi.')
    df=bills()
    if df.empty or not (df.sisa>0).any():st.info('Tidak ada tagihan yang perlu dibayar.')
    else:
        pending=df[df.sisa>0]
        opts={f"#{r.id} — {r.nama} — {r.jenis} {r.periode} — sisa {rupiah(r.sisa)}":int(r.id) for r in pending.itertuples()}
        selected=st.selectbox('Pilih tagihan',list(opts));tid=opts[selected]
        row=pending[pending.id==tid].iloc[0]
        st.info(f"Tagihan {rupiah(row['nominal'])} | Dibayar {rupiah(row['dibayar'])} | Sisa {rupiah(row['sisa'])}")
        with st.form('pay'):
            tanggal=st.date_input('Tanggal pembayaran',value=date.today())
            nominal=st.number_input('Jumlah dibayar (Rp)',min_value=1,max_value=int(row['sisa']),value=int(row['sisa']))
            metode=st.selectbox('Metode',['Tunai','Transfer','Lainnya']);catatan=st.text_input('Catatan')
            if st.form_submit_button('Simpan pembayaran'):
                with conn() as c:
                    sisa=c.execute('SELECT t.nominal-COALESCE((SELECT SUM(p.nominal) FROM pembayaran p WHERE p.tagihan_id=t.id),0) FROM tagihan t WHERE t.id=?',(tid,)).fetchone()[0]
                    if nominal>sisa:st.error('Jumlah melebihi sisa tagihan. Muat ulang halaman.')
                    else:
                        c.execute('INSERT INTO pembayaran(tagihan_id,tanggal,nominal,metode,catatan) VALUES(?,?,?,?,?)',(tid,tanggal.isoformat(),int(nominal),metode,catatan));c.commit();st.success('Pembayaran berhasil dicatat.');st.rerun()
    pay=read('''SELECT p.id,p.tanggal,s.nis,s.nama,s.kelas,t.jenis,t.periode,p.nominal,p.metode,p.catatan FROM pembayaran p JOIN tagihan t ON t.id=p.tagihan_id JOIN siswa s ON s.id=t.siswa_id ORDER BY p.id DESC''')
    st.subheader('Riwayat Transaksi')
    st.dataframe(pay,hide_index=True,use_container_width=True)
    if not pay.empty:
        download_csv(pay,'Unduh transaksi CSV','pembayaran.csv')
        pid=st.selectbox('Pilih nomor bukti transaksi',pay.id.tolist())
        r=pay[pay.id==pid].iloc[0]
        receipt=f'''BUKTI PEMBAYARAN — ANUGERAH INSANI\nNomor: AI-{int(r['id']):06d}\nTanggal: {r['tanggal']}\nNIS: {r['nis']}\nNama: {r['nama']}\nKelas: {r['kelas']}\nTagihan: {r['jenis']} / {r['periode']}\nJumlah: {rupiah(r['nominal'])}\nMetode: {r['metode']}\nCatatan: {r['catatan']}\n\nBukti dibuat dari catatan aplikasi lokal.'''
        st.code(receipt)
        st.download_button('Unduh bukti pembayaran (.txt)',receipt.encode('utf-8'),f'bukti_AI_{pid:06d}.txt')

elif menu=='Analisis & Laporan':
    hero('Analisis & Laporan','Jelajahi tren pemasukan dan tunggakan dengan grafik interaktif.')
    df=bills();pay=read('SELECT tanggal,nominal FROM pembayaran ORDER BY tanggal')
    if df.empty:st.info('Belum ada data untuk dianalisis.')
    else:
        a,b=st.columns(2)
        with a:
            st.plotly_chart(style_chart(px.bar(df.groupby('kelas',as_index=False)[['nominal','dibayar','sisa']].sum(),x='kelas',y=['nominal','dibayar','sisa'],barmode='group',title='Rekap per kelas')),use_container_width=True)
        with b:
            if not pay.empty:
                pay['bulan']=pay.tanggal.str.slice(0,7)
                st.plotly_chart(style_chart(px.line(pay.groupby('bulan',as_index=False).nominal.sum(),x='bulan',y='nominal',markers=True,title='Tren penerimaan per bulan')),use_container_width=True)
            else:st.info('Belum ada transaksi.')
        st.subheader('Tunggakan jatuh tempo')
        overdue=df[df.status=='Tunggakan']
        st.dataframe(overdue[['nis','nama','kelas','jenis','periode','jatuh_tempo','sisa']],hide_index=True,use_container_width=True)
        download_csv(df,'Unduh laporan lengkap CSV','laporan_keuangan.csv')

elif menu=='Prediksi Machine Learning':
    hero('Prediksi Machine Learning','Analisis keterlambatan, kesiapan dataset, dan prediksi risiko berbasis riwayat.','INTELLIGENT INSIGHTS')
    st.caption('Model tidak menganggap tagihan yang belum jatuh tempo sebagai terlambat. Data yang belum cukup tetap bisa dianalisis tanpa membuat skor risiko semu.')
    df=bills()
    if df.empty:
        st.warning('Belum ada tagihan. Impor dataset melalui menu Upload Dataset Excel.')
    else:
        today=pd.Timestamp(date.today())
        hist=df.copy()
        hist['due']=pd.to_datetime(hist.jatuh_tempo,errors='coerce').dt.normalize()
        hist['last']=pd.to_datetime(hist.terakhir_bayar,errors='coerce').dt.normalize()
        # Tagihan hanya berlabel jika sudah jatuh tempo, atau sudah lunas.
        mature=hist[(hist['due']<today) | ((hist.sisa==0)&hist['last'].notna())].copy()
        mature['late']=((mature['due']<today)&(mature.sisa>0)) | ((mature.sisa==0)&(mature['last']>mature['due']))
        mature=mature.sort_values(['siswa_id','due','id'])
        mature['prior_late']=mature.groupby('siswa_id')['late'].transform(lambda x:x.shift().expanding().mean()).fillna(0)
        mature['prior_count']=mature.groupby('siswa_id').cumcount()
        mature['due_month']=mature.due.dt.month
        nlate=int(mature.late.sum()); nontime=len(mature)-nlate
        upcoming=int((hist['due']>=today).sum())
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Tagihan total',len(hist))
        c2.metric('Historis dapat dinilai',len(mature))
        c3.metric('Terlambat',nlate)
        c4.metric('Tepat waktu',nontime)
        st.subheader('Analisis status tagihan')
        col1,col2=st.columns([1,1])
        with col1:
            counts=df.status.value_counts().rename_axis('Status').reset_index(name='Jumlah')
            st.plotly_chart(px.pie(counts,names='Status',values='Jumlah',hole=.6,title='Distribusi status saat ini'),use_container_width=True)
        with col2:
            byclass=df.groupby('kelas',as_index=False).agg(sisa=('sisa','sum')).sort_values('sisa',ascending=False).head(15)
            st.plotly_chart(px.bar(byclass,x='kelas',y='sisa',title='Sisa tagihan per kelas',labels={'sisa':'Rupiah','kelas':'Kelas'}),use_container_width=True)
        st.subheader('Kesiapan pelatihan Random Forest')
        st.subheader('Eksperimen Random Forest menggunakan 500 siswa')
        st.info('Mode demonstrasi: sistem membuat 6 bulan riwayat pembayaran SINTETIS untuk siswa yang sudah diimpor. Data asli dan tanggal transaksi asli tidak diubah. Hasilnya bukan prediksi risiko nyata.')
        if st.button('Jalankan Random Forest — data simulasi historis',type='primary'):
            st.session_state['ml_demo_run']=True
        if st.session_state.get('ml_demo_run',False):
            history=make_demo_history(df)
            if history.empty:
                st.warning('Impor dataset siswa terlebih dahulu.')
            else:
                model_demo,report_demo=train_demo(history)
                st.success(f'Model demo berhasil dilatih dengan {len(history):,} catatan historis sintetis dari {history.siswa_id.nunique()} siswa.'.replace(',','.'))
                a,b,c,d=st.columns(4)
                a.metric('Akurasi uji',f"{report_demo['akurasi']:.1%}")
                b.metric('Presisi terlambat',f"{report_demo['presisi']:.1%}")
                c.metric('Recall terlambat',f"{report_demo['recall']:.1%}")
                d.metric('F1-score',f"{report_demo['f1']:.1%}")
                st.caption('Evaluasi dilakukan pada bulan simulasi terakhir yang tidak digunakan untuk pelatihan. Angka hanya berlaku untuk pola sintetis.')
                st.plotly_chart(px.bar(pd.DataFrame({'Kelas':['Tepat waktu','Terlambat'],'Jumlah':[(history.late==0).sum(),(history.late==1).sum()]}),x='Kelas',y='Jumlah',title='Distribusi label sintetis'),use_container_width=True)
                upcoming_demo=df[(df.sisa>0)&(pd.to_datetime(df.jatuh_tempo)>=today)].copy()
                if not upcoming_demo.empty:
                    scored=score_current(model_demo,history,upcoming_demo)
                    st.subheader('Skor demonstrasi untuk tagihan dataset 500 siswa')
                    st.warning('SKOR SIMULASI — bukan penilaian kemampuan membayar, tidak untuk menagih atau memberi sanksi.')
                    st.dataframe(scored[['nis','nama','kelas','periode','sisa','skor_simulasi','kategori_simulasi']],hide_index=True,use_container_width=True)
                    st.download_button('Unduh skor simulasi CSV',scored[['nis','nama','kelas','periode','sisa','skor_simulasi','kategori_simulasi']].to_csv(index=False).encode('utf-8-sig'),'skor_simulasi_500_siswa.csv','text/csv')
                with st.expander('Lihat data historis sintetis'):
                    st.dataframe(history[['nis','kelas','due','tanggal_bayar','nominal','late','prior_late']],use_container_width=True,hide_index=True)
                    st.download_button('Unduh riwayat simulasi CSV',history.to_csv(index=False).encode('utf-8-sig'),'riwayat_simulasi_random_forest.csv','text/csv')
        st.divider()
        st.subheader('Model dari data pembayaran asli')
        enough=len(mature)>=20 and nlate>=5 and nontime>=5
        if not enough:
            st.warning('Prediksi belum diaktifkan: diperlukan minimal 20 tagihan yang dapat dinilai, termasuk 5 terlambat dan 5 tepat waktu.')
            st.progress(min(1.,len(mature)/20),text=f'Jumlah historis: {len(mature)} / 20')
            st.write(f'Contoh terlambat: **{nlate}/5** | Tepat waktu: **{nontime}/5** | Tagihan dengan jatuh tempo hari ini atau mendatang: **{upcoming}**')
            st.info('Solusi: unggah riwayat pembayaran dari bulan-bulan sebelumnya, dengan tanggal jatuh tempo dan tanggal pembayaran yang benar. Jangan mengubah tanggal atau label hanya agar model bisa dilatih.')
            with st.expander('Mengapa model belum bisa dilatih?',expanded=True):
                st.write('Model klasifikasi memerlukan variasi label. Apabila seluruh contoh tepat waktu, model tidak bisa mempelajari karakteristik keterlambatan. Untuk tagihan yang belum jatuh tempo, status akhirnya juga belum diketahui.')
            st.download_button('Unduh data diagnostik CSV',mature[['nis','nama','kelas','jatuh_tempo','terakhir_bayar','nominal','dibayar','late']].to_csv(index=False).encode('utf-8-sig'),'diagnostik_ml.csv','text/csv')
        else:
            features=['nominal','due_month','prior_late','prior_count']
            X=mature[features]; y=mature.late.astype(int)
            # Validasi berdasarkan waktu menghindari penggunaan informasi masa depan dalam pengujian.
            ordered=mature.sort_values(['due','id'])
            split=int(len(ordered)*.8)
            train=ordered.iloc[:split]; test=ordered.iloc[split:]
            if train.late.nunique()<2 or test.late.nunique()<2:
                st.warning('Data kedua kategori tersedia, tetapi pembagian pelatihan/pengujian berdasarkan waktu belum memiliki kedua kategori. Tambahkan riwayat dari beberapa periode untuk evaluasi yang lebih andal.')
            else:
                model=RandomForestClassifier(n_estimators=180,min_samples_leaf=3,random_state=42,class_weight='balanced')
                model.fit(train[features],train.late.astype(int))
                predicted=model.predict(test[features])
                st.metric('Akurasi pada periode pengujian',f'{accuracy_score(test.late.astype(int),predicted):.1%}')
                with st.expander('Laporan evaluasi model'):
                    st.text(classification_report(test.late.astype(int),predicted,zero_division=0))
                    st.dataframe(pd.DataFrame({'Fitur':features,'Kontribusi relatif':model.feature_importances_}),hide_index=True)
                # Hanya tagihan yang belum jatuh tempo; tagihan yang telah terlambat bukan prediksi.
                current=df[(df.sisa>0)&(pd.to_datetime(df.jatuh_tempo)>=today)].copy()
                if current.empty:
                    st.info('Tidak ada tagihan terbuka dengan jatuh tempo mendatang untuk diprediksi.')
                else:
                    stats=mature.groupby('siswa_id').agg(prior_late=('late','mean'),prior_count=('late','size')).reset_index()
                    current=current.merge(stats,on='siswa_id',how='left').fillna({'prior_late':0,'prior_count':0})
                    current['due_month']=pd.to_datetime(current.jatuh_tempo).dt.month
                    model.fit(X,y)
                    current['skor_model']=(model.predict_proba(current[features])[:,list(model.classes_).index(1)]*100).round(1)
                    current['kategori']=pd.cut(current.skor_model,bins=[-1,35,70,101],labels=['Rendah','Sedang','Tinggi'])
                    st.subheader('Estimasi risiko untuk tagihan mendatang')
                    st.dataframe(current[['nis','nama','kelas','periode','sisa','jatuh_tempo','skor_model','kategori']],hide_index=True,use_container_width=True)
                    st.download_button('Unduh prediksi CSV',current.to_csv(index=False).encode('utf-8-sig'),'prediksi_risiko.csv','text/csv')
                st.caption('Skor Random Forest belum dikalibrasi sebagai probabilitas nyata. Jangan gunakan hasilnya untuk sanksi atau keputusan otomatis terhadap siswa.')

elif menu=='Upload Dataset Excel':
    hero('Upload Dataset Excel','Impor data siswa, tagihan, dan pembayaran dari workbook Excel ke dashboard secara otomatis.','DATA INTEGRATION')
    st.info('Format yang didukung: sheet Data_Siswa, Data_Tagihan, dan Data_Pembayaran, seperti dataset 500 siswa yang Anda berikan. Sheet Dashboard dihitung ulang dari database.')
    up=st.file_uploader('Pilih file dataset Excel (.xlsx)',type=['xlsx'],help='Unggah workbook monitoring pembayaran sekolah.')
    if up is not None:
        try:
            raw=up.getvalue()
            dataset=read_excel(raw)
            summary=validate(dataset)
            st.success('Struktur dataset valid dan siap diimpor.')
            a,b,c=st.columns(3)
            a.metric('Siswa pada file',f"{summary['siswa']:,}")
            b.metric('Tagihan pada file',f"{summary['tagihan']:,}")
            c.metric('Transaksi pada file',f"{summary['pembayaran']:,}")
            d,e=st.columns(2)
            d.metric('Total nilai tagihan',rupiah(summary['total_tagihan']))
            e.metric('Total pembayaran',rupiah(summary['total_bayar']))
            st.markdown('#### Pratinjau dataset')
            tab1,tab2,tab3=st.tabs(['👥 Data Siswa','🧾 Data Tagihan','💳 Data Pembayaran'])
            for tab,name in zip((tab1,tab2,tab3),('Data_Siswa','Data_Tagihan','Data_Pembayaran')):
                with tab:
                    st.dataframe(dataset[name].head(100),use_container_width=True,hide_index=True)
                    st.caption(f"Menampilkan maksimum 100 dari {len(dataset[name])} baris.")
            existing=sum(int(read(f'SELECT COUNT(*) AS n FROM {t}').iloc[0]['n']) for t in ('siswa','tagihan','pembayaran'))
            st.markdown('#### Konfirmasi impor')
            replace=False
            if existing:
                st.warning('Database saat ini sudah berisi data. Impor penuh akan mengganti seluruh siswa, tagihan, dan pembayaran yang tersimpan.')
                replace=st.checkbox('Saya sudah membuat cadangan dan setuju mengganti SELURUH data lama.')
            else:
                st.caption('Database kosong. Dataset dapat langsung dimasukkan.')
            if st.button('📥 Impor Dataset ke Database',type='primary',disabled=bool(existing and not replace)):
                result=import_data(DB,dataset,replace=bool(existing))
                st.success(f"Berhasil mengimpor {result['siswa']} siswa, {result['tagihan']} tagihan, dan {result['pembayaran']} transaksi.")
                st.balloons()
                st.info('Pilih menu Dashboard untuk melihat hasil impor dan menu Prediksi Machine Learning untuk analisis.')
        except Exception as exc:
            st.error(f'Dataset belum dapat diimpor: {exc}')
            st.caption('Pastikan semua ID konsisten, nominal positif, tanggal valid, dan pembayaran tidak melebihi tagihan.')

elif menu=='Data Simulasi':
    hero('Data Simulasi','Uji seluruh fitur menggunakan data fiktif yang terpisah dari data nyata.','DEMO ENVIRONMENT')
    st.write('Membuat 40 siswa fiktif, 6 bulan tagihan SPP, dan transaksi contoh dengan pola keterlambatan bervariasi. Tidak menggunakan data siswa asli.')
    st.warning('Data simulasi hanya boleh ditambahkan pada database kosong agar tidak tercampur dengan data operasional.')
    existing=read('SELECT COUNT(*) n FROM siswa').iloc[0]['n']
    if existing:st.info(f'Database telah berisi {existing} siswa. Tombol demo dinonaktifkan.')
    if st.button('Buat 40 siswa dan riwayat simulasi',disabled=bool(existing)):
        rng=np.random.default_rng(27)
        with conn() as c:
            for i in range(1,41):
                nis=f'DEMO{i:04d}';nama=f'Siswa Simulasi {i:02d}';kelas=f'{1+(i-1)//10}A'
                cur=c.execute('INSERT INTO siswa(nis,nama,kelas,wali,kontak) VALUES(?,?,?,?,?)',(nis,nama,kelas,'Wali Simulasi',''))
                sid=cur.lastrowid
                tendency=(i%8)/10
                for month in range(5,11):
                    due=date(2026,month,10)
                    amount=200000+(i%4)*25000
                    cur2=c.execute('INSERT INTO tagihan(siswa_id,jenis,periode,nominal,jatuh_tempo) VALUES(?,?,?,?,?)',(sid,'SPP',f'2026-{month:02d}',amount,due.isoformat()))
                    tid=cur2.lastrowid
                    late=bool(rng.random()<tendency)
                    pay_date=due+timedelta(days=int(rng.integers(2,22))) if late else due-timedelta(days=int(rng.integers(0,8)))
                    # All demo bills are paid; labels are derived from actual simulated payment dates.
                    c.execute('INSERT INTO pembayaran(tagihan_id,tanggal,nominal,metode,catatan) VALUES(?,?,?,?,?)',(tid,pay_date.isoformat(),amount,'Transfer','DATA SIMULASI'))
            c.commit()
        st.success('Data simulasi berhasil dibuat. Buka Dashboard atau Prediksi Machine Learning.');st.rerun()
