from OFET_mobility_prediction import *
import argparse
import pickle
from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import rdchem
import optuna
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
import os
import math
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors, rdMolDescriptors

def Train_uC_Nfeatures(fea,N,file_path):
    target=['uC*','Vth(eV)','u_h','u_e']
    others=['U_IDX','Smiles']
    items=list(fea.keys())
    #print(type(items))
    for x in target+others:
        try:
            items.remove(x)
        except:
            pass
    combinations = list(itertools.combinations(items,N))
    
    uhC=[]
    ueC=[]
    for i in range(len(fea['uC*'])):
        if fea['u_h'][i]!=0 and not np.isnan(fea['u_h'][i]):
            uhC.append(fea['uC*'][i])
        else:
            uhC.append(0)
    for i in range(len(fea['uC*'])):
        if fea['u_e'][i]!=0 and not np.isnan(fea['u_e'][i]):
            ueC.append(fea['uC*'][i])
        else:
            ueC.append(0)
    outcome_h,outcome_e={},{}
    if not os.path.exists(file_path+f'uhC_{N}features'):
        os.makedirs(file_path+f'uhC_{N}features')
    if not os.path.exists(file_path+f'ueC_{N}features'):
        os.makedirs(file_path+f'ueC_{N}features')
    for ii in range(math.ceil(len(combinations)/10000)):
        features_h,R2_h=[],[]
        features_e,R2_e=[],[]
        print(ii)
        for i in tqdm(range(10000*ii,10000*ii+10000)):
            if i<len(combinations):  
                comb=combinations[i]
                df_FT=pd.DataFrame(index=[i for i in range(len(fea['u_h']))],columns=comb)
                for key in comb:
                    df_FT[key]=fea[key]
                df_FT['uhC']=uhC
                df_FT=df_FT.drop_duplicates().reset_index().drop(['index'],axis=1)
                x_h_train,y_h_train,x_h_test,y_h_test,NN=Data_split(42,0.9,df_FT.iloc[:,:-1].values,df_FT['uhC'].values,log=True)
                X_train, X_val, y_train, y_val = x_h_train,x_h_test,y_h_train,y_h_test
                model = XGBRegressor(random_state=42)
                model.fit(X_train, y_train)
                val_r2_h = cross_val_score(model, X_train+X_val, y_train+y_val, cv=3, scoring='r2').mean()
                features_h.append(comb)
                R2_h.append(val_r2_h)
                df_FT=pd.DataFrame(index=[i for i in range(len(fea['u_e']))],columns=comb)
                for key in comb:
                    df_FT[key]=fea[key]
                df_FT['ueC']=ueC
                df_FT=df_FT.drop_duplicates().reset_index().drop(['index'],axis=1)
                x_e_train,y_e_train,x_e_test,y_e_test,NN=Data_split(42,0.9,df_FT.iloc[:,:-1].values,df_FT['ueC'].values,log=True)
                X_train, X_val, y_train, y_val = x_e_train,x_e_test,y_e_train,y_e_test
                model = XGBRegressor(random_state=42)
                model.fit(X_train, y_train)
                val_r2_e = cross_val_score(model, X_train+X_val, y_train+y_val, cv=3, scoring='r2').mean()
                features_e.append(comb)
                R2_e.append(val_r2_e)
            else:
                break
        outcome_h['R2']=R2_h
        outcome_h['Features']=features_h
        np.save(file_path+f'ueC_{N}features/part_{ii}.npy',outcome_h)
        outcome_e['R2']=R2_e
        outcome_e['Features']=features_e
        np.save(file_path+f'uhC_{N}features/part_{ii}.npy',outcome_e)

def Get_F_Part(path,save_file=None):
    files=os.listdir(path)
    features,R2=[],[]
    step=0.01
    items=[]
    for file in files:
        data=np.load(path+file,allow_pickle=True).item()
        features=features+data['Features']
        R2=R2+data['R2']
    MIN,MAX=round(min(R2),2),round(max(R2),2)
    N=int((1-MIN)/step)
    for i in range(len(features)):
        items+=features[i]
    items=list(set(items))
    items.sort()
    ini_mat=[[0 for i in range(len(items))] for j in range(N)]
    print(items)
    df=pd.DataFrame(ini_mat,index=[i for i in range(N)],columns=items)
    df['Total_Num']=[0 for i in range(N)]
    df['R2']=[f'{MIN+i*step}-{MIN+(i+1)*step}' for i in range(df.shape[0])]
    N_fea={}
    for i in tqdm(range(len(R2))):
        idx=int((R2[i]-MIN)//step)
        for fea in features[i]:
            df[fea][:idx+1]+=1
            if fea not in N_fea:
                N_fea[fea]=1
            else:
                N_fea[fea]+=1
    for col in df.columns.values[:-2]:
        df[col]=df[col]/N_fea[col]
    if save_file:
        df.to_csv(save_file,index=None)
    else:
        pass
    return df

def Get_Polymer_Info(mols):
    df=pd.DataFrame(index=[i for i in range(len(mols))],columns=['Mass','HAcceptors','HDonors','LogP','TPSA','RingCount','NumRotatableBonds','NumHeteroatoms'])
    for i in range(len(mols)):
        try:
            m = Chem.AddHs(mols[i])
            df['Mass'][i] = np.round(Descriptors.MolWt(m),1)
            df['LogP'][i] = np.round(Descriptors.MolLogP(m),2)
            df['HDonors'][i] = rdMolDescriptors.CalcNumLipinskiHBD(m)
            df['HAcceptors'][i] = rdMolDescriptors.CalcNumLipinskiHBA(m)
            df['TPSA'][i] = np.round(Descriptors.TPSA(m),1)
            df['NumRotatableBonds'][i]= rdMolDescriptors.CalcNumRotatableBonds(m)
            df['RingCount'][i] = m.GetRingInfo().NumRings()
            df['NumHeteroatoms'][i] = sum(1 for atom in m.GetAtoms() if atom.GetAtomicNum() not in (1, 6))
        except:
            print(i)
    return df


def Get_Units_Info(Unit_list,INFO_df):
    U_INFO={}
    U_INFO['Delta_HOMO'],U_INFO['Delta_LUMO'],U_INFO['VAR_HOMO'],U_INFO['VAR_LUMO'],U_INFO['AVE_HOMO'],U_INFO['AVE_LUMO']=[],[],[],[],[],[]
    U_INFO['AVE_LOLIPOP'],U_INFO['VAR_LOLIPOP']=[],[]
    for i in range(41):
        E=round(-7.0+0.1*i,1)
        U_INFO[f'DOS{E}']=[]
    for i in range(len(Unit_list)):
        HOMO,LUMO,Mass,LO=[],[],[],[]
        x=Unit_list[i]
        for j in x:
            HOMO.append(INFO_df['HOMO (eV)'][j])
            LUMO.append(INFO_df['LUMO (eV)'][j])
            Mass.append(INFO_df['Mass'][j])
            LO.append(INFO_df['LOLIPOP'][j])
        delta_H=[abs(HOMO[0]-HOMO[-1])]
        delta_L=[abs(LUMO[0]-LUMO[-1])]
        Mass_par=[mass/sum(Mass) for mass in Mass]
        for i in range(41):
            E=round(-7.0+0.1*i,1)
            d=0
            for j in range(len(x)):
                d+=Mass_par[j]*INFO_df[f'DOS{E}'][x[j]]
            U_INFO[f'DOS{E}'].append(d)
        for j in range(len(x)-1):
            delta_H.append(abs(HOMO[j]-HOMO[j+1]))
            delta_L.append(abs(LUMO[j]-LUMO[j+1]))
        U_INFO['Delta_HOMO'].append(np.mean(delta_H))
        U_INFO['Delta_LUMO'].append(np.mean(delta_L))
        U_INFO['VAR_HOMO'].append(np.var(HOMO))
        U_INFO['VAR_LUMO'].append(np.var(LUMO))
        U_INFO['VAR_LOLIPOP'].append(np.var(LO))
        U_INFO['AVE_LOLIPOP'].append(np.mean(LO))
        U_INFO['AVE_HOMO'].append(np.mean(HOMO))
        U_INFO['AVE_LUMO'].append(np.mean(LUMO))
    return pd.DataFrame.from_dict(U_INFO)

def Get_fea_data(OECT_file,Units_file,features,targets,module,OFET_h_Path,OFET_e_Path,OFETs_fea_name):
    FT=Preporcess_File(OECT_file,features,targets)
    U_IDX=[]
    for i in range(FT.shape[0]):
        IDX=[int(x) for x in FT['U_index'][i][1:-1].split(',')]
        U_IDX.append(IDX)
    FT['U_IDX']=U_IDX
    FT2=Get_OFET_Features(FT,features,targets,module)
    OFETs_fea=FT2[OFETs_fea_name]
    if os.path.isdir(OFET_h_Path):
        P_files=os.listdir(OFET_h_Path)
        uh=np.zeros(FT2.shape[0])
        for OFET_h_m in P_files:
            #print(OFET_h_m)
            model=pickle.load(open(OFET_h_Path+OFET_h_m, "rb"))
            uh+=model.predict(OFETs_fea.values)/len(P_files)
        FT2['uh_pred']=uh
    if os.path.isdir(OFET_e_Path):
        E_files=os.listdir(OFET_e_Path)
        ue=np.zeros(FT2.shape[0])
        for OFET_e_m in E_files:
            #print(OFET_h_m)
            model=pickle.load(open(OFET_e_Path+OFET_e_m, "rb"))
            ue+=model.predict(OFETs_fea.values)/len(E_files)
        FT2['ue_pred']=ue
    if not os.path.isdir(OFET_h_Path):
        model=pickle.load(open(OFET_h_Path, "rb"))
        FT2['uh_pred']=model.predict(OFETs_fea.values)
    if not os.path.isdir(OFET_e_Path):
        model2=pickle.load(open(OFET_e_Path, "rb"))
        FT2['ue_pred']=model2.predict(OFETs_fea.values)
    FT2['U_IDX']=[[0,0] for i in range(FT2.shape[0])]
    FT2['Smiles']=['a' for i in range(FT2.shape[0])]
    for i in range(FT2.shape[0]):
        ii=FT['HOMO(eV)'].values.tolist().index(FT2['HOMO(eV)'][i])
        FT2['U_IDX'][i]=FT['U_IDX'][ii]
        FT2['Smiles'][i]=FT['poly1_smiles'][ii]
    df=pd.read_csv(Units_file).iloc[:,1:]
    U_INFO=Get_Units_Info(FT2['U_IDX'].values.tolist(),df)
    fea={}
    mols=[Chem.MolFromSmiles(mm) for mm in FT2['Smiles']]
    P_INFO=Get_Polymer_Info(mols)
    for name in FT2.columns.values:
        fea[name]=FT2[name]
    for key,value in U_INFO.items():
        fea[key]=value
    for key,value in P_INFO.items():
        fea[key]=value
    return fea
    
def Get_dataset_feature(file,HL_module,COS2_module,Units_file,OFET_features,OFET_h_Path,OFET_e_Path):#'module/UnimolHLsave_INPUT'
    data=np.load(file,allow_pickle=True).item()
    df=pd.DataFrame(data['Polymer_smile'])
    df.columns=['SMILES']
    df.to_csv('Pol.csv',index=None)
    pred_m=MolPredict(load_model=HL_module)
    pred_HL=pred_m.predict('Pol.csv')
    os.remove("Pol.csv")
    feat=Get_COS2_Feature(data['Dimer_smiles'],COS2_module,save_file=None,N_dihe=10,Fea_Enhance=False).iloc[:,:-1]
    feat['HOMO(eV)']=[x[0] for x in pred_HL]
    feat['LUMO(eV)']=[x[1]-0.8 for x in pred_HL]
    feat['Mn(kg/mol)']=[30 for x in range(feat.shape[0])]
    feat['PDI']=[2 for x in range(feat.shape[0])]
    feat['SC_INFO1']=[0 for x in range(feat.shape[0])]
    feat['SC_INFO2']=[0 for x in range(feat.shape[0])]
    feat['SC_INFO3']=[0 for x in range(feat.shape[0])]
    mols=[Chem.MolFromSmiles(mm) for mm in df['SMILES']]
    P_INFO=Get_Polymer_Info(mols)
    df=pd.read_csv(Units_file)
    U_INFO=Get_Units_Info(data['Index'],df)
    for key,value in U_INFO.items():
        feat[key]=value
    for key,value in P_INFO.items():
        feat[key]=value
        
    if os.path.isdir(OFET_h_Path):
        P_files=os.listdir(OFET_h_Path)
        uh=np.zeros(feat.shape[0])
        for OFET_h_m in P_files:
                #print(OFET_h_m)
            model=pickle.load(open(OFET_h_Path+OFET_h_m, "rb"))
            uh+=model.predict(feat[OFET_features].values)/len(P_files)
        feat['uh_pred']=uh
    if os.path.isdir(OFET_e_Path):
        E_files=os.listdir(OFET_e_Path)
        ue=np.zeros(feat.shape[0])
        for OFET_e_m in E_files:
                #print(OFET_h_m)
            model=pickle.load(open(OFET_e_Path+OFET_e_m, "rb"))
            ue+=model.predict(feat[OFET_features].values)/len(E_files)
        feat['ue_pred']=ue
    if not os.path.isdir(OFET_h_Path):
        model=pickle.load(open(OFET_h_Path, "rb"))
        feat['uh_pred']=model.predict(feat[OFET_features].values)
    if not os.path.isdir(OFET_e_Path):
        model2=pickle.load(open(OFET_e_Path, "rb"))
        feat['ue_pred']=model2.predict(feat[OFET_features].values)
    return feat

