from rdkit import Chem
from tqdm import tqdm
from itertools import product
import pandas as pd
import numpy as np

def Get_id_bysymbol(combo,symbol):#获取symbol标记的元素序号
    for at in combo.GetAtoms():
        if at.GetSymbol()==symbol:
            return at.GetIdx()

def Get_neiid_bysymbol(combo,symbol):#获取symbol相邻的第一个的元素序号
    for at in combo.GetAtoms():
        if at.GetSymbol()==symbol:
            at_nei=at.GetNeighbors()[0]
            return at_nei.GetIdx()
        
def combine2frag(Amol,symbol1,Bmol,symbol2):#Fr和Cs表记位置连接
    combo = Chem.CombineMols(Amol,Bmol)
    Symbol1_NEI_ID=Get_neiid_bysymbol(combo,symbol1)
    Symbol2_NEI_ID=Get_neiid_bysymbol(combo,symbol2)
    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(Symbol1_NEI_ID,Symbol2_NEI_ID,order=Chem.rdchem.BondType.SINGLE)

    ID1=Get_id_bysymbol(combo,symbol1)
    edcombo.RemoveAtom(ID1)
    back = edcombo.GetMol()


    ID2=Get_id_bysymbol(back,symbol2)

    edcombo=Chem.EditableMol(back)
    edcombo.RemoveAtom(ID2)
    back = edcombo.GetMol()
    smi= Chem.MolToSmiles(back)
    return smi

def Get_U(U,symbol1,symbol2):#讲单元的smiles描述符转化为无连符号的smiles
    U1=U.replace(symbol1,'H')
    U2=U1.replace(symbol2,'H')
    m=Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(U2)))
    return m

def Generate_Poly_NUnits(mol_list,N):#N<=5
    sym=[['Fr','Cs'],['Am','Cm'],['Pm','Sm'],['Ba','Ra'],['Ar','Kr']]
    Poly=[]
    Frags=[]
    U_IDX=[]
    if N==2:
        print('Generating 2-Units polymers')
        order_list=[x for x in product(sym[0],sym[1])]
        for i in tqdm(range(len(mol_list))):
            for j in range(i,len(mol_list)):
                molA=mol_list[i]
                molB=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[j]).replace('Fr','Am').replace('Cs','Cm'))
                for sym_idx in range(int(len(order_list)/2)):
                    frags=[]
                    frags.append(combine2frag(molA,order_list[len(order_list)-sym_idx-1][0],molB,order_list[sym_idx][1]))
                    frags.append(combine2frag(molA,order_list[sym_idx][0],molB,order_list[len(order_list)-sym_idx-1][1]))
                    Frags.append(frags)
                    Poly.append(combine2frag(molA,order_list[len(order_list)-sym_idx-1][0],molB,order_list[sym_idx][1]))
                    U_IDX.append([i,j])
                #print(i,j,Frags[-1])
    if N==3:
        print('Generating 3-Units polymers')
        order_list=[ x for x in product(sym[0],sym[1],sym[2])]
        for i in range(len(mol_list)):
            for j in range(i,len(mol_list)):
                for k in range(j,len(mol_list)):
                    molA=mol_list[i]
                    molB=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[j]).replace('Fr','Am').replace('Cs','Cm'))
                    molC=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[k]).replace('Fr','Pm').replace('Cs','Sm'))
                    for sym_idx in range(int(len(order_list)/2)):
                        frags=[]
                        dimer1=combine2frag(molA,order_list[len(order_list)-sym_idx-1][0],molB,order_list[sym_idx][1])
                        frags.append(dimer1)
                        dimer2=combine2frag(molB,order_list[len(order_list)-sym_idx-1][1],molC,order_list[sym_idx][2])
                        frags.append(dimer2)
                        dimer3=combine2frag(molA,order_list[sym_idx][0],molC,order_list[len(order_list)-sym_idx-1][2])
                        frags.append(dimer3)
                        poly=combine2frag(Chem.MolFromSmiles(dimer1),order_list[len(order_list)-sym_idx-1][1],molC,order_list[sym_idx][2])
                        Frags.append(frags)
                        Poly.append(poly)
                        U_IDX.append([i,j,k])
    if N==4:
        print('Generating 4-Units polymers')
        order_list=[x for x in product(sym[0],sym[1],sym[2],sym[3])]
        for i in range(len(mol_list)):
            for j in range(i,len(mol_list)):
                for k in range(j,len(mol_list)):
                    for s in range(k,len(mol_list)):
                        molA=mol_list[i]
                        molB=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[j]).replace('Fr','Am').replace('Cs','Cm'))
                        molC=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[k]).replace('Fr','Pm').replace('Cs','Sm'))
                        molD=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[s]).replace('Fr','Ba').replace('Cs','Ra'))
                        for sym_idx in range(int(len(order_list)/2)):
                            frags=[]
                            dimer1=combine2frag(molA,order_list[len(order_list)-sym_idx-1][0],molB,order_list[sym_idx][1])
                            dimer2=combine2frag(molB,order_list[len(order_list)-sym_idx-1][1],molC,order_list[sym_idx][2])
                            dimer3=combine2frag(molC,order_list[len(order_list)-sym_idx-1][2],molD,order_list[sym_idx][3])
                            dimer4=combine2frag(molA,order_list[sym_idx][0],molD,order_list[len(order_list)-sym_idx-1][3])
                            frags.append(dimer1)
                            frags.append(dimer2)
                            frags.append(dimer3)
                            frags.append(dimer4)
                            trimer1=combine2frag(Chem.MolFromSMiles(dimer1),order_list[len(order_list)-sym_idx-1][1],molC,order_list[sym_idx][2])
                            poly=combine2frag(Chem.MolFromSmiles(trimer1),order_list[len(order_list)-sym_idx-1][2],molD,order_list[sym_idx][3])
                            Poly.append(poly)
                            Frags.append(frags)
                            U_IDX.append([i,j,k,s])
    if N==5:
        print('Generating 5-Units polymers')
        order_list=[x for x in product(sym[0],sym[1],sym[2],sym[3],sym[4])]
        for i in range(len(mol_list)):
            for j in range(i,len(mol_list)):
                for k in range(j,len(mol_list)):
                    for s in range(k,len(mol_list)):
                        for t in range(s,len(mol_list)):
                            molA=mol_list[i]
                            molB=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[j]).replace('Fr','Am').replace('Cs','Cm'))
                            molC=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[k]).replace('Fr','Pm').replace('Cs','Sm'))
                            molD=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[s]).replace('Fr','Ba').replace('Cs','Ra'))
                            molE=Chem.MolFromSmiles(Chem.MolToSmiles(mol_list[t]).replace('Fr','Ar').replace('Cs','Kr'))
                            for sym_idx in range(int(len(order_list)/2)):
                                frags=[]
                                dimer1=combine2frag(molA,order_list[len(order_list)-sym_idx-1][0],molB,order_list[sym_idx][1])
                                dimer2=combine2frag(molB,order_list[len(order_list)-sym_idx-1][1],molC,order_list[sym_idx][2])
                                dimer3=combine2frag(molC,order_list[len(order_list)-sym_idx-1][2],molD,order_list[sym_idx][3])
                                dimer4=combine2frag(molD,order_list[len(order_list)-sym_idx-1][3],molE,order_list[sym_idx][4])
                                dimer5=combine2frag(molA,order_list[sym_idx][0],molE,order_list[len(order_list)-sym_idx-1][4])
                                frags.append(dimer1)
                                frags.append(dimer2)
                                frags.append(dimer3)
                                frags.append(dimer4)
                                frags.append(dimer5)
                                Frags.append(frags)
                                trimer1=combine2frag(Chem.MolFromSmiles(dimer1),order_list[len(order_list)-sym_idx-1][1],molC,order_list[sym_idx][2])
                                quad=combine2frag(Chem.MolFromSmiles(trimer1),order_list[len(order_list)-sym_idx-1][2],molD,order_list[sym_idx][3])
                                poly=combine2frag(Chem.MolFromSmiles(quad),order_list[len(order_list)-sym_idx-1][3],molE,order_list[sym_idx][4])
                                Poly.append(poly)
                                U_IDX.append([i,j,k,s,t])
    
    npy_save={'Polymer_smile':Poly,'Dimer_smiles':Frags,'Index':U_IDX}
    np.save(f'Generate_{N}units_unprocess.npy',npy_save)
    #print(Poly,Frags,U_IDX)
    Poly2,Frags2,U_IDX2=[],[],[]
    for i in tqdm(range(len(Poly))):
        x=Frags[i]
        x2=[]
        p=Poly[i]
        for cha in ['Fr','Cs','Am','Cm','Pm','Sm','Ba','Ra','Ar','Kr']:
            if cha in Poly[i]:
                p=p.replace(cha,'H')
        p=Chem.MolToSmiles(Chem.MolFromSmiles(p))
        for j in range(len(x)):
            u=x[j]
            for cha in ['Fr','Cs','Am','Cm','Pm','Sm','Ba','Ra','Ar','Kr']:
                if cha in u:
                    u=u.replace(cha,'H')
            x2.append(Chem.MolToSmiles(Chem.MolFromSmiles(u)))
        if len(set(U_IDX[i]))==1:
            x2=[x2[0]]
            p=x2[0]
            U_IDX[i]=[U_IDX[i][0]]
        if p not in Poly2:
            Frags2.append(x2)
            U_IDX2.append(U_IDX[i])
            Poly2.append(p)
            print(x2,U_IDX[i],p,Poly[i])
        if i%1000==0:
            print(f'We are collecting combined molecular: {i}/{len(Poly)}')
    npy_save={'Polymer_smile':Poly2,'Dimer_smiles':Frags2,'Index':U_IDX2}
    np.save(f'Generate_{N}units.npy',npy_save)
    print(len(Frags2),len(Frags))
    return Poly2,Frags2,U_IDX2

def main():
    U_smiles=pd.read_csv('../data/Collected_Units.csv')['smiles']
    U_mols=[Chem.MolFromSmiles(x) for x in U_smiles]
    poly0,frags0,IDX0=Generate_Poly_NUnits(U_mols,2)
    #poly1,frags1,IDX1=Generate_Poly_NUnits(U_mols,3)

if __name__ == "__main__":
    main()
