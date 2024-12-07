import pandas as pd
import os
from Units_Generation import Get_id_bysymbol
from rdkit import Chem
from rdkit.Chem import AllChem,rdmolops
import numpy as np
from tqdm import tqdm
import math
from unimol_tools import MolTrain, MolPredict
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import argparse
import re
import random
from time import time

def Preporcess_File(file,features,targets):
    data=pd.read_csv(file)
    df=data.drop_duplicates(subset=['poly1_smiles','poly2_smiles'], keep='first', inplace=False, ignore_index=False).drop(features+targets,axis=1)
    df_mean=data.groupby(['poly1_smiles','poly2_smiles'])[features+targets].mean().reset_index()
    df2=pd.merge(df_mean,df,on=['poly1_smiles','poly2_smiles'])
    #print(df_mean.shape,df2.shape,df2.columns)
    idx=[]
    for i in range(df2.shape[0]):
        condition=[pd.isna(df2[feature][i])==False for feature in features]#+[pd.isna(df2[tar][i])==False for tar in targets]
        if all(condition) and df2['class'][i]=='HP':
            idx.append(i)#特征完全且是HP类型分子
    FT=df2.iloc[idx,:].reset_index()
    drop_name=[]
    for i in range(FT.shape[1]):
        #print(FT.columns.values[i],FT.columns.values[i] not in features+targets,not isinstance(FT.iloc[0][i],str))
        if FT.columns.values[i] not in features+targets and not isinstance(FT.iloc[0][i],str):
            drop_name.append(FT.columns.values[i])
            
    FT=FT.drop(drop_name,axis=1)
    return FT

def Get_COS2(x1):#计算（N，19）形式数据的cos2
    PHI=[]
    PP=[]
    if len(x1[0])>1:
        x=x1
    if len(x1[0])==1:
        x=[]
        for i in range(int(len(x1)/19)):
            a=[]
            for j in range(19):
                a.append(x1[19*i+j])
            x.append(a)
    for i in range(len(x)):
        P=[]
        phi=0
        for j in range(19):
            P.append(math.exp(-x[i][j]/(1.987072*0.001*298)))
        A=1/sum(P)
        ang=[math.cos(math.pi*ii/18) for ii in range(19)]
        for ii in range(len(P)):
            P[ii]=P[ii]*A
            phi+=(ang[ii]**2)*P[ii]
        PP.append(P)
        PHI.append(phi)
    return PHI
    
def SMI_COS2_Prediction(smi0,module,save_file=None):#smi0为待预测dimer_list，smi_name为保存的文件名称,module为加载的模型名称
    smi=[]
    for x in smi0:
        smi+=x
    smi=list(set(smi))   
    data_smi=pd.DataFrame(index=[i for i in range(len(smi))],columns=['SMILES']+[f'TARGET_{i+1}' for i in range(19)])
    data_smi['SMILES']=smi
    for i in range(19):
        data_smi[f'TARGET_{i+1}']=0
    CT = time()
    data_smi.to_csv(f'Dimer_prediction{CT}.csv',index=None)
    predm = MolPredict(load_model=module)
    pred_y = predm.predict(f'Dimer_prediction{CT}.csv')
    os.remove(f'Dimer_prediction{CT}.csv')
    COS2=Get_COS2(pred_y)
    for i in range(len(pred_y)):
        for j in range(len(pred_y[0])):
            data_smi[f'TARGET_{j+1}'][i]=pred_y[i][j]
    if save_file:
        data_smi['COS2']=COS2
        data_smi.to_csv(save_file,index=None)
    else:
        pass
    U_COS2=[]
    for i in range(len(smi0)):
        cos2=[]
        for x in smi0[i]:
            idx=smi.index(x)
            cos2.append(COS2[idx])
        U_COS2.append(cos2)
    return U_COS2

def Get_COS2_Feature(smi0,module,save_file=None,N_dihe=10,Fea_Enhance=True):
    U_COS2=SMI_COS2_Prediction(smi0,module,save_file=None)
    IDX=[]
    New_COS2=[]
    if Fea_Enhance:
        for i in range(len(U_COS2)):
            cos2=[]
            x=U_COS2[i]
            for j in range(len(x)):
                if x[j:]+x[:j] not in cos2:
                    cos2.append(x[j:]+x[:j])
                    New_COS2.append(x[j:]+x[:j])
                    IDX.append(i)
                if x[::-1][j:]+x[::-1][:j] not in cos2:
                    cos2.append(x[::-1][j:]+x[::-1][:j])
                    New_COS2.append(x[::-1][j:]+x[::-1][:j])
                    IDX.append(i)
    else:
        New_COS2=U_COS2
    COS2_Fea=pd.DataFrame(index=[i for i in range(len(New_COS2))],columns=[f'COS2-{i}' for i in range(N_dihe)]+['poly_index'])
    if IDX:
        COS2_Fea['poly_index']=IDX
    else:
        COS2_Fea['poly_index']=[i for i in range(len(New_COS2))]
    for i in range(len(New_COS2)):
        for j in range(N_dihe):
            JJ=j%len(New_COS2[i])
            COS2_Fea.iloc[i,j]=New_COS2[i][JJ]
    return COS2_Fea
    
def Get_Mol(mol,c):#原分子mol，获取的原子序号list：c，创建原子集合c组成的新分子
    #mol=Chem.AddHs(mol)
    atom_indices = c  # 定义一个包含要提取的原子的标号列表（无需按连接顺序）
    editable_mol = Chem.EditableMol(Chem.Mol())# 创建一个可编辑的分子对象
    old_to_new_index = {}# 创建一个字典，用于映射旧原子索引到新原子索引
    for old_index in atom_indices:# 添加所需的原子到可编辑的分子中，并记录映射
        #print(old_index)
        atom = mol.GetAtomWithIdx(old_index)
        new_index = editable_mol.AddAtom(atom)
        old_to_new_index[old_index] = new_index
    for bond in mol.GetBonds():# 添加所需的键到可编辑的分子中
        if bond.GetBeginAtomIdx() in atom_indices and bond.GetEndAtomIdx() in atom_indices:
            begin_atom_index = old_to_new_index[bond.GetBeginAtomIdx()]
            end_atom_index = old_to_new_index[bond.GetEndAtomIdx()]
            bond_type = bond.GetBondType()
            editable_mol.AddBond(begin_atom_index, end_atom_index, bond_type)
    result_mol = editable_mol.GetMol()# 获取最终的可编辑分子
    result_smiles = Chem.MolToSmiles(result_mol)# 将分子转换为SMILES字符串以进行可视化或其他操作
    
    return result_mol
    
def Get_dihes(mol):#获取mol中可旋转二面角
    #pytmol label atom rank
    dihes=[]
    for bond in mol.GetBonds():
        if bond.GetBondType().name=='SINGLE' and bond.IsInRing()==False:#寻找不在环上的单键
            atom2id=bond.GetBeginAtomIdx()
            atom3id=bond.GetEndAtomIdx()
            atom2=bond.GetBeginAtom()
            atom3=bond.GetEndAtom()

            if len(atom3.GetNeighbors())==1  or len(atom2.GetNeighbors())==1 :
                pass
            else:
                atom1s=atom2.GetNeighbors()
                atom1sid=[ at.GetIdx()   for at in atom1s]
    #             print(atom1sid,atom3id)
                atom1sid.remove(atom3id)
                atom1id=atom1sid[0]

                atom4s=atom3.GetNeighbors()
                atom4sid=[ at.GetIdx()   for at in atom4s]
    #             print(atom1sid,atom3id)
                atom4sid.remove(atom2id)
                atom4id=atom4sid[0]
#                 print(atom1id,atom2id,atom3id,atom4id)
                dihe=[atom1id,atom2id,atom3id,atom4id]
                dihes.append(dihe)
    return dihes
    
def Dfs_paths(graph, start, end, path=[]):#从start到end的所有路径
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for atom in graph[start]:
        if atom not in path:
            new_paths = Dfs_paths(graph, atom, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths

def Get_graph(mol):#获取分子连接图
    graph={}
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        if atom1 not in graph:
            graph[atom1] = []
        if atom2 not in graph:
            graph[atom2] = []
        graph[atom1].append(atom2)
        graph[atom2].append(atom1)
    return graph
    
def Get_path_atoms(m,Mark1,Mark2):#获取连接Mark1 Mark2之间的共轭路径
    start_atom_index = Get_id_bysymbol(m,Mark1) # 第一个碳原子
    target_atom_index = Get_id_bysymbol(m,Mark2)  # 氧原子，索引为 -1 表示最后一个原子
    graph=Get_graph(m)
    # 获取所有路径
    paths = Dfs_paths(graph, start_atom_index, target_atom_index)
    ATOMS=[]
    for path in paths:
        #print(path)
        for atom in path:
            if atom not in ATOMS:
                ATOMS.append(atom)
    return ATOMS# 输出所有路径上的原子
    
def Get_poly_backbone(m,M1,M2):#获取分子共轭骨架
    ssr = Chem.GetSymmSSSR(m)
    RING_Atom=[]
    idx=Get_path_atoms(m,M1,M2)
    for r_L in ssr:
        RING_Atom.append(list(r_L))
    idx2=[]#获取相邻环
    for bond in m.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        for i in range(len(RING_Atom)):
            if (atom1 in idx and atom2 in RING_Atom[i]) or (atom2 in idx and atom1 in RING_Atom[i]):
                for j in range(len(RING_Atom[i])):
                    idx2.append(RING_Atom[i][j])
            else:
                pass
    idx3=[]
    for i in range(len(RING_Atom)):
        set_c = set(RING_Atom[i]) & set(idx2)
        list_c = list(set_c)
        if list_c and list_c!=RING_Atom[i]:
            for j in range(len(RING_Atom[i])):
                idx3.append(RING_Atom[i][j])
    IDX=[]
    for id in [idx,idx2,idx3]:
        for j in id:
            if j not in IDX:
                IDX.append(j)
    #print(IDX)
    IDX_atoms=[m.GetAtomWithIdx(x) for x in IDX]#周围官能团：
    IDX2=[x for x in IDX]
    for atom in IDX_atoms:
        for atom2 in atom.GetNeighbors():
            if atom2.GetIdx() not in IDX:
                IDX.append(atom2.GetIdx())
            for atom3 in atom2.GetNeighbors():
                if atom3.GetSymbol()!='C':
                    IDX.append(atom3.GetIdx())
    for atom in IDX_atoms:
        for atom2 in atom.GetNeighbors():
            if atom2.GetIdx() not in IDX2 and atom2.GetSymbol()!='C':
                IDX2.append(atom2.GetIdx())
            nei=[]
            for nei1 in atom2.GetNeighbors():
                #print(nei1 in IDX_atoms,nei1.GetIdx())
                if nei1.GetIdx() not in [a.GetIdx() for a in IDX_atoms]:
                    nei.append(nei1)
            #print([atom.GetIdx() for atom in nei])
            if atom2.GetIdx() not in IDX2 and atom2.GetSymbol()=='C' and all(atom3.GetSymbol()!='C' for atom3 in nei):
                IDX2.append(atom2.GetIdx())
                #print('zz',atom2.GetIdx())
                for atom3 in atom2.GetNeighbors():
                    IDX2.append(atom3.GetIdx())
                     #print(atom3.GetIdx())
    IDX3=[]
    IDX4=[]
    for i in range(len(IDX)):
        if IDX[i] not in IDX3:
            IDX3.append(IDX[i])
    for i in range(len(IDX2)):
        if IDX2[i] not in IDX4:
            IDX4.append(IDX2[i])            
    return IDX3,IDX4#前面为包含亚甲基，后面为不包含亚甲基的部分

def Decomp_Poly(m,M1,M2,CH3=False):#分解为骨架和侧链
    IDX,IDX2=Get_poly_backbone(m,M1,M2)
    if CH3:
        BB=Chem.MolFromSmiles(Chem.MolToSmiles(Get_Mol(m,IDX)))#取代基位置的根据化合价自动加氢
        try:
            SC=Chem.GetMolFrags(AllChem.ReplaceCore(m, Get_Mol(m,IDX)),asMols=True)
        except:
            SC=[]
    else:
        BB=Chem.MolFromSmiles(Chem.MolToSmiles(Get_Mol(m,IDX2)))#取代基位置的根据化合价自动加氢
        try:
            SC=Chem.GetMolFrags(AllChem.ReplaceCore(m, Get_Mol(m,IDX2)),asMols=True)
        except:
            SC=[]
    return BB,SC


def Get_Not_C(mol):#获取侧链非碳原子数目
    N=0
    for atom in mol.GetAtoms():
        if atom.GetSymbol()!='C' and atom.GetSymbol()!='*':
            N+=1
    return N

def Get_Bifurcation_site(mol,mark):#获取分叉位点
    BEG_IDX=Get_id_bysymbol(mol,mark)
    #print(BEG_IDX)
    Len=[]
    for atom in mol.GetAtoms():
    # 检查当前原子是否为碳原子
        if atom.GetAtomicNum() == 6:  # 6 是碳原子的原子序数
            # 检查当前碳原子是否连接到至少一个非氢原子并且是SP3杂化
            if atom.GetTotalNumHs()==1 and atom.GetHybridization()==Chem.HybridizationType.SP3:
                BS=atom.GetIdx()
                path=rdmolops.GetShortestPath(mol, BEG_IDX, BS)
                Len.append(len(path)-1)
    return Len

def Get_SC_INF(m,M1,M2):
    BB,SC=Decomp_Poly(m,M1,M2)
    if SC:
        SMI=[Chem.MolToSmiles(x) for x in SC]
        SMI2=[]
        #print(SMI)
        for smi in SMI:
            stri=re.match('\[(\d+)\*\]',smi)[0]
            smi=smi.replace(stri,'[*]')
            SMI2.append(smi)
        mms=[Chem.MolFromSmiles(x) for x in SMI2]
        nc=[]
        for i in range(len(mms)):
            #mark=re.match(r'\[\d+\*\]', Chem.MolToSmiles(SC[i]))[0]
            N=Get_Not_C(mms[i])
            nc.append(N)
        b=np.sum(nc)
        smi=list(set([Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in SMI2]))#侧链去重，获取侧链种类
        SC=[Chem.MolFromSmiles(x) for x in smi]
        INFO=[]
        #print(smi)
        for i in range(len(SC)):
            #mark=re.match(r'\[\d+\*\]', Chem.MolToSmiles(SC[i]))[0]
            N=Get_Not_C(SC[i])
            Len=Get_Bifurcation_site(SC[i],'*')
            if len(Len)!=0:
                INFO.append([len(Len),Len[0],N,SC[i].GetNumAtoms()])
            else:
                INFO.append([0,0,N,SC[i].GetNumAtoms()])
        a=np.sort(np.array([x[1] for x in INFO]))[::-1]#排序按照分叉位点从多到少
        #总杂原子数目
        BS=[0,0,b]#只考虑分叉位点前两名的侧链。为0表示侧链没有分叉位点，为+N表示侧链在N号位置分叉，为-1表示没有侧链。
        for i in range(len(a)):
            try:
                BS[i]=a[i]
            except:
                pass
    else:
        BS=[-1,-1,0]
    return BS

def Get_OFET_Features(exp_file,features,target,COS2_module,SC_INFO=True,M1='Fr',M2='Cs',save_file=None,N_dihe=10,Fea_Enhance=True):
    print('Preprocessing files')
    if isinstance(exp_file,str):
        data=Preporcess_File(exp_file,features,target)
    if isinstance(exp_file,pd.DataFrame):
        data=exp_file
    smi0=[]
    for i in range(data.shape[0]):
    #print(i)
        smis1=data['dimers1'][i].split("'")
        dimer=[]
        for j in range(len(smis1)):
            if j%2==1:
                dimer.append(Chem.MolToSmiles(Chem.MolFromSmiles(smis1[j]),canonical=True))
        smi0.append(dimer)
    print('Predicting the planarity index with Uni-mol model')
    U_COS2=Get_COS2_Feature(smi0,COS2_module,save_file,N_dihe,Fea_Enhance)
    
    print('Side chain information is colleted:')
    if SC_INFO:
        feat_name=features+[f'COS2-{i}' for i in range(N_dihe)]+['SC_INFO1','SC_INFO2','SC_INFO3']+target
        SC_INFO=[]
        for i in tqdm(range(data.shape[0])):
            m=Chem.MolFromSmiles(data['poly1_smiles'][i])
            SC_INFO.append(Get_SC_INF(m,M1,M2))
        data['SC_INFO1']=[x[0] for x in SC_INFO]
        data['SC_INFO2']=[x[1] for x in SC_INFO]
        data['SC_INFO3']=[x[2] for x in SC_INFO]
    else:
        feat_name=features+[f'COS2-{i}' for i in range(N_dihe)]+target
    print('Side chain information has been colleted')
    Features=pd.DataFrame(index=[i for i in range(U_COS2.shape[0])],columns=feat_name)
    print(f'Follow features are preocessing: {feat_name}')
    for name in [f'COS2-{i}' for i in range(N_dihe)]:
        Features[name]=U_COS2[name]
    for name in features+target:
        for i in range(U_COS2.shape[0]):
            IDX=U_COS2['poly_index'][i]
            Features[name][i]=data[name][IDX]
            if SC_INFO:
                Features['SC_INFO3'][i]=data['SC_INFO3'][IDX]
                Features['SC_INFO1'][i]=data['SC_INFO1'][IDX]
                Features['SC_INFO2'][i]=data['SC_INFO2'][IDX]
    Features=Features.drop_duplicates().reset_index().drop(['index'],axis=1)
    print('OFET features have been colleted')
    return Features

def Data_split(random_seed,ratio,x,y,log=True):
    X,Y=[],[]
    for i in range(len(y)):
        if y[i]!=0 and not np.isnan(y[i]):
            X.append(x[i])
            Y.append(y[i])
            
    data_array=np.arange(0,len(X),1)
    np.random.seed(random_seed)
    np.random.shuffle(data_array)
    train_x,train_y,test_x,test_y=[],[],[],[]
    train_L=int(ratio*len(data_array))
    N=[]
    if log:
        for i in data_array[:train_L]:
            train_y.append(np.log(Y[i]))
            train_x.append(X[i])
            N.append(i)
        for i in data_array[train_L:]:
            test_y.append(np.log(Y[i]))
            test_x.append(X[i])
            N.append(i)
    else:
        for i in data_array[:train_L]:
            train_y.append(Y[i])
            train_x.append(X[i])
            N.append(i)
        for i in data_array[train_L:]:
            test_y.append(Y[i])
            test_x.append(X[i])
            N.append(i)
    return train_x,train_y,test_x,test_y,N

def Train(train_x,train_y,test_x,test_y):
    rs=42
    regressors = [
        ('Lasso Regressor',Lasso(random_state=rs)),
        ('RlasticNet Regressor',ElasticNet(random_state=rs)),#random_state=rs
        ('PolynomilaFeatures',make_pipeline(PolynomialFeatures(4), LinearRegression())),
        ("Linear Regression", LinearRegression()), # 线性回归模型
        ("Ridge Regression", Ridge(random_state=rs)), # 岭回归模型
        ("Support Vector", SVR()),  # 支持向量回归模型
        ("K-Nearest Neighbors", KNeighborsRegressor()),  # K-最近邻回归模型
        ("Decision Tree", DecisionTreeRegressor(random_state=rs)),  # 决策树回归模型
        ("Random Forest", RandomForestRegressor(random_state=rs)), # 随机森林回归模型
        ("Gradient Boosting", GradientBoostingRegressor(random_state=rs)), # 梯度提升回归模型
        ("XGBoost", XGBRegressor(random_state=rs)), # XGBoost回归模型
        ("LightGBM", LGBMRegressor(random_state=rs)), # LightGBM回归模型
        ("Multi-layer Perceptron", MLPRegressor( # 多层感知器（神经网络）回归模型
            hidden_layer_sizes=(128,64,32),
            learning_rate_init=0.0001,
            activation='relu', solver='adam',
            batch_size=16,
            max_iter=10000, random_state=rs)),
    ]
    R2={}
    Pred_y={}
    for name, regressor in regressors:
        regressor.fit(train_x,train_y)
        pred_train_y = regressor.predict(train_x)
        pred_test_y = regressor.predict(test_x)
        R2[name]=r2_score(pred_test_y,test_y)
        Pred_y[name]=pred_test_y
        mse=mean_squared_error(pred_test_y,test_y)
        print(name,mse,R2[name])
    return R2,Pred_y

def Objective(trial):
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    reg_alpha=trial.suggest_float('reg_alpha', 1e-5,100, log=True)
    reg_lambda=trial.suggest_float('reg_lambda', 1e-5,100, log=True)
    gamma=trial.suggest_float('gamma', 0, 0.5,step=0.1)
    model = XGBRegressor(max_depth=max_depth,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         gamma=gamma,
                         objective='reg:squarederror')
    r2 = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
    return r2
    
def parse_args():
    parser = argparse.ArgumentParser(description='OFET mobility prediction')
    parser.add_argument('--file_path', type=str, default='../data/OFET.csv',help='Path for storing the experimental data file')
    #parser.add_argument('--con_sym2', type=str, default='Cs',help='Connected Atom_2 symbol')
    parser.add_argument('--OFET_features', type=list, default=['Mn(kg/mol)','PDI','HOMO(eV)','LUMO(eV)'],help='Experimental Features')
    parser.add_argument('--Consider_sidechain', type=bool, default=True, help='Whether to consider side chain information')
    parser.add_argument('--Unimol_HOMOLUMO_model',type=str,default='../module/Unimolsave',help='Path for storing Unimol HOMO and LUMO prediction model')
    parser.add_argument('--Junction_site',type=list,default=['Fr','Cs'],help='Symbol of Junction site')
    parser.add_argument('--Save_Unimol_predictionfile',default=None,help='Whether to save the Unimol model prediction results file')
    parser.add_argument('--N_dihedral_angle',type=int,default=10,help='Number of dihedral angle in polymer')
    parser.add_argument('--Data_Enhancement',type=bool,default=True,help='Whether to perform data enhancement')
    parser.add_argument('--OFET_targets',type=list,default=['u_h','u_e'],help='Prediction targets')
    parser.add_argument('--OFET_model_save_PT',type=str,default='../module/OFET_prediction/',help='Path for OFETs property prediction model')
    args = parser.parse_args([])
    return args
    
def Plot_XY(x,y,color,size,x_min=None,x_max=None,grid=True,tick=None,diag=True,diag_line='dashed',save_name=None,other_plot_dashed=None,other_plot_solid=None):
    plt.style.use('fast')
    plt.figure(figsize=(9,9))
    plt.scatter(x,y,c=color,s=size)
    if diag:
        plt.plot([-100,100],[-100,100],color='black',linestyle=diag_line)
    plt.tick_params(axis='x',labelsize=30,colors='black')
    plt.tick_params(axis='y',labelsize=30,colors='black')
    MIN=min([min(x),min(y)])
    MAX=max([max(x),max(y)])
    if other_plot_dashed is not None:
        for x in other_plot_dashed:
            plt.plot(x[0],x[1],c='black',linestyle='dashed')
    if other_plot_solid is not None:
        for x in other_plot_solid:
            plt.plot(x[0],x[1],c='black',linestyle='solid')            
    if not x_min:
        plt.xlim(round(MIN-(MAX-MIN)*0.1),round(MAX+(MAX-MIN)*0.1))
        plt.ylim(round(MIN-(MAX-MIN)*0.1),round(MAX+(MAX-MIN)*0.1))
    if x_min:
        plt.xlim(x_min,x_max)
        plt.ylim(x_min,x_max)
    if tick is None:
        tick=np.linspace(round(MIN),round(MAX),5)
    plt.xticks(tick)
    plt.yticks(tick)
    plt.grid(grid)
    if save_name:
        plt.savefig(save_name)
    plt.show()

if __name__ == "__main__":
    args=parse_args()
    args.OFET_features=['HOMO(eV)','LUMO(eV)']#'Mn(kg/mol)','PDI',
    args.Consider_sidechain=False
    features=Get_OFET_Features(args.file_path,args.OFET_features,
                               args.OFET_targets,args.Unimol_HOMOLUMO_model,
                               SC_INFO=args.Consider_sidechain,
                               M1=args.Junction_site[0],M2=args.Junction_site[1],
                               save_file=args.Save_Unimol_predictionfile,
                               N_dihe=args.N_dihedral_angle,
                               Fea_Enhance=args.Data_Enhancement)
    if args.Consider_sidechain:
       Fea_name=args.OFET_features+['COS2-0','COS2-1','COS2-2','COS2-3','COS2-4','COS2-5','COS2-6','COS2-7','COS2-8','COS2-9','SC_INFO1','SC_INFO2','SC_INFO3'] 
    if not args.Consider_sidechain:
       Fea_name=args.OFET_features+['COS2-0','COS2-1','COS2-2','COS2-3','COS2-4','COS2-5','COS2-6','COS2-7','COS2-8','COS2-9']
    RS = np.random.choice(1000,20)
    features['u']=[0 for i in range(features.shape[0])]
    m_h,m_e=[],[]
    Pre_PT=args.OFET_model_save_PT
    for i in range(20):
        x_h_train,y_h_train,x_h_test,y_h_test,N=Data_split(RS[i],0.9,features[Fea_name].values.tolist(),features['u_h'].values.tolist(),log=True)
        x_e_train,y_e_train,x_e_test,y_e_test,N=Data_split(RS[i],0.9,features[Fea_name].values.tolist(),features['u_e'].values.tolist(),log=True)
        print(len(x_h_train+x_h_test),len(x_e_train+x_e_test))
        N=len(x_h_train[0])
        X_train, X_val, y_train, y_val = x_h_train,x_h_test,y_h_train,y_h_test
        #a,b=Train(x_h_train,y_h_train,x_h_test,y_h_test)
        print('The parameters of the OFET hole mobility prediction model are being optimized')
        m0=XGBRegressor(random_state=42)
        m0.fit(X_train,y_train)
        val_preds=m0.predict(X_test)
        val_r2 = r2_score(y_val, val_preds)
        if val_r2>0.85 and len(m_h)<5:
            study = optuna.create_study(direction='maximize')
            study.optimize(Objective, n_trials=200)
            best_params = study.best_params
            final_model = XGBRegressor(max_depth=best_params['max_depth'],
                                           learning_rate=best_params['learning_rate'],
                                           n_estimators=best_params['n_estimators'],
                                           reg_alpha=best_params['reg_alpha'],
                                           reg_lambda=best_params['reg_lambda'],
                                           gamma=best_params['gamma'],
                                           objective='reg:squarederror')
                
            final_model.fit(X_train, y_train)
            m_h.append(final_model)
            val_preds = final_model.predict(X_val)
            val_r2 = r2_score(y_val, val_preds)
            print('Validation R²:', val_r2)
            if not os.path.exists(Pre_PT+f'Features_{N}D_hole'):
                os.makedirs(Pre_PT+f'Features_{N}D_hole')
            pickle.dump(final_model, open(Pre_PT+f"Features_{N}D_hole/model_OFET_hole_mobility_{i}.dat","wb"))
        X_train, X_val, y_train, y_val = x_e_train,x_e_test,y_e_train,y_e_test
        #a,b=Train(x_e_train,y_e_train,x_e_test,y_e_test)
        print('The parameters of the OFET electron mobility prediction model are being optimized')
        m0=XGBRegressor(random_state=42)
        m0.fit(X_train,y_train)
        val_preds=m0.predict(X_test)
        val_r2 = r2_score(y_val, val_preds)
        if val_r2>0.85 and len(m_e)<5:
            study = optuna.create_study(direction='maximize')
            study.optimize(Objective, n_trials=200)
            best_params = study.best_params
            final_model = XGBRegressor(max_depth=best_params['max_depth'],
                                           learning_rate=best_params['learning_rate'],
                                           n_estimators=best_params['n_estimators'],
                                           reg_alpha=best_params['reg_alpha'],
                                           reg_lambda=best_params['reg_lambda'],
                                           gamma=best_params['gamma'],
                                           objective='reg:squarederror')
                
            final_model.fit(X_train, y_train)
            m_e.append(final_model)
            val_preds = final_model.predict(X_val)
            val_r2 = r2_score(y_val, val_preds)
            print('Validation R²:', val_r2)
            if not os.path.exists(Pre_PT+f'Features_{N}D_electron'):
                os.makedirs(Pre_PT+f'Features_{N}D_electron')
            pickle.dump(final_model, open(Pre_PT+f"Features_{N}D_electron/model_OFET_electron_mobility_{i}.dat","wb"))