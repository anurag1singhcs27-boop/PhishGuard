import re
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).parent
DATA = ROOT 
st.set_page_config(page_title='PhishGuard | Phishing Detection', page_icon='🛡️', layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv(DATA/'email_samples.csv'), pd.read_csv(DATA/'url_samples.csv')

def url_features(raw):
    raw = str(raw).strip()
    p = urlparse(raw if '://' in raw else 'http://' + raw)
    host = p.netloc.lower().split('@')[-1].split(':')[0]
    return {'length':len(raw),'dots':raw.count('.'),'hyphens':raw.count('-'),'ats':raw.count('@'),
            'questions':raw.count('?'),'equals':raw.count('='),'slashes':raw.count('/'),
            'digits':sum(c.isdigit() for c in raw),'subdomains':max(0,len(host.split('.'))-2),
            'https':int(p.scheme.lower()=='https'),'ip_host':int(bool(re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}',host))),
            'path_length':len(p.path),'query_length':len(p.query),'punycode':int('xn--' in host)}

def metrics(y, pred):
    return {'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,pos_label='phishing',zero_division=0),
            'Recall':recall_score(y,pred,pos_label='phishing',zero_division=0),'F1-score':f1_score(y,pred,pos_label='phishing',zero_division=0),
            'Confusion matrix':confusion_matrix(y,pred,labels=['legitimate','phishing']).tolist()}

@st.cache_resource
def train():
    emails, urls = load_data()
    ex, ey = emails.text.astype(str), emails.label.astype(str)
    ux = pd.DataFrame([url_features(u) for u in urls.url])
    uy = urls.label.astype(str)
    email_model = Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=5000)),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))]).fit(ex,ey)
    url_model = Pipeline([('scale',StandardScaler()),('model',RandomForestClassifier(n_estimators=150,class_weight='balanced',random_state=42))]).fit(ux,uy)
    extr, exte, eytr, eyte = train_test_split(ex,ey,test_size=.25,random_state=42,stratify=ey)
    et = Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),max_features=5000)),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))]).fit(extr,eytr)
    utr, ute, ytr, yte = train_test_split(ux,uy,test_size=.25,random_state=42,stratify=uy)
    ut = Pipeline([('scale',StandardScaler()),('model',RandomForestClassifier(n_estimators=150,class_weight='balanced',random_state=42))]).fit(utr,ytr)
    return email_model,url_model,{'Email NLP — TF-IDF + Logistic Regression':metrics(eyte,et.predict(exte)),'URL features — Random Forest':metrics(yte,ut.predict(ute))}

st.markdown('''<div style="padding:24px;border-radius:16px;background:linear-gradient(110deg,#102a43,#176b87);color:white"><h1 style="color:white;margin:0">🛡️ PhishGuard</h1><p style="margin:8px 0 0">Hybrid ML–NLP prototype for phishing email and URL classification</p></div>''',unsafe_allow_html=True)
st.write('Check email text or URL structure for a first-level model classification. This prototype is educational and does not guarantee that a message or website is safe.')
with st.sidebar:
    st.header('Analysis settings')
    mode=st.radio('Choose input type',['Email analysis','URL analysis'])
    st.divider(); st.caption('Payal Jha · Anurag Singh'); st.caption('Project Guide: Dr. Saurabh Kumar')
try:
    email_model,url_model,eval_metrics=train()
except Exception as e:
    st.error(f'Model setup failed: {e}'); st.stop()
left,right=st.columns([1.2,.8],gap='large')
with left:
    st.subheader('Analyse input')
    if mode=='Email analysis':
        sample=st.selectbox('Example',['Write my own','Suspicious verification email','Normal meeting email'])
        examples={'Suspicious verification email':'URGENT: Your account will be suspended. Click here to verify your password immediately.','Normal meeting email':'Hi team, the meeting agenda is attached. Please review it before tomorrow’s project meeting.'}
        text=st.text_area('Email subject/body (do not paste real passwords or private information)',value=examples.get(sample,''),height=180,placeholder='Paste email text here…')
        if st.button('Analyse email',type='primary',use_container_width=True):
            if len(text.strip())<8: st.warning('Enter a little more text to analyse.')
            else:
                pred=email_model.predict([text])[0]; probs=email_model.predict_proba([text])[0]; classes=list(email_model.classes_)
                st.session_state.result=(pred,float(probs[classes.index('phishing')]),'email')
    else:
        sample=st.selectbox('Example',['Write my own','Suspicious URL example','Documentation URL'])
        examples={'Suspicious URL example':'http://secure-login-account-verify.example.com/signin?session=123','Documentation URL':'https://docs.python.org/3/library/'}
        url=st.text_input('URL',value=examples.get(sample,''),placeholder='https://example.com/account')
        if st.button('Analyse URL',type='primary',use_container_width=True):
            if len(url.strip())<4 or ' ' in url.strip(): st.warning('Enter a URL without spaces.')
            else:
                x=pd.DataFrame([url_features(url)]); pred=url_model.predict(x)[0]; probs=url_model.predict_proba(x)[0]; classes=list(url_model.classes_)
                st.session_state.result=(pred,float(probs[classes.index('phishing')]),'url')
with right:
    st.subheader('How the system works')
    st.markdown('**Email:** text cleaning → TF-IDF → Logistic Regression → prediction\n\n**URL:** URL parsing → lexical/structural features → Random Forest → prediction')
    st.info('Separate models are used because email text and URL structure require different features.')
if 'result' in st.session_state:
    pred,score,kind=st.session_state.result; st.divider(); st.subheader('Analysis result')
    if pred=='phishing': st.error('⚠️ Model classification: PHISHING')
    else: st.success('Model classification: LEGITIMATE')
    st.metric('Model-estimated phishing score',f'{score*100:.1f}%')
    st.caption('This score is not a calibrated real-world probability. A “Legitimate” result does not prove that the content is safe.')
with st.expander('Model evaluation — demonstration samples'):
    st.warning('Metrics below are calculated on a held-out split of the small illustrative datasets bundled with this starter. Do not present them as real-world performance. Replace the samples with properly sourced datasets before final evaluation.')
    st.dataframe(pd.DataFrame([{'Pipeline':name,**{k:round(v,3) for k,v in m.items() if k!='Confusion matrix'}} for name,m in eval_metrics.items()]),use_container_width=True,hide_index=True)
    for name,m in eval_metrics.items():
        st.write('**'+name+' — confusion matrix** (rows: actual legitimate/phishing; columns: predicted legitimate/phishing)')
        st.dataframe(pd.DataFrame(m['Confusion matrix'],index=['Actual legitimate','Actual phishing'],columns=['Predicted legitimate','Predicted phishing']),use_container_width=True)
with st.expander('Dataset and limitations'):
    e,u=load_data(); a,b=st.columns(2); a.metric('Email demo records',len(e)); b.metric('URL demo records',len(u))
    st.write('The included CSVs are illustrative starter samples, not a representative public cybersecurity dataset. This prototype does not inspect webpage content, follow redirects or query threat-intelligence services.')
st.caption('Python · Pandas · Scikit-learn · Streamlit · GitHub')
