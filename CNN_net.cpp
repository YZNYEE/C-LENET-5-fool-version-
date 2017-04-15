#include"Matrix.h"
#include"CNN_net.h"
#include<windows.h>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<fstream>
using namespace std;

CNN::CNN(int i,int c,int s,int kernel,int scale,int *num)
{
    if(num==nullptr)
    {
        cout<<"请输入inputmap的大小"<<endl;
        cin>>inputmap;
        cout<<"请输入C层的数量"<<endl;
        cin>>NumOfC;
        cout<<"请输入S层的数量"<<endl;
        cin>>NumOfS;
        cout<<"请输入Kernel的大小"<<endl;
        cin>>Kernel;
        cout<<"请输入Scale的大小"<<endl;
        cin>>Scale;

        this->NumMapOfC=new int[NumOfC];
        for(int i=0;i<NumOfC;i++)
        {
            cout<<"请输入C"<<i<<"层map的数量"<<endl;
            cin>>NumMapOfC[i];
        }
    }
    else{
        inputmap=i;
        NumOfC=c;
        NumOfS=s;
        Kernel=kernel;
        Scale=scale;
        this->NumMapOfC=new int[NumOfC];
        for(int i=0;i<NumOfC;i++)
            NumMapOfC[i]=num[i];
    }
    C=new Matrix**[NumOfC];
    CFeature=new Matrix**[NumOfC];
    CKernel=new Matrix***[NumOfC];
    CKGradient=new Matrix***[NumOfC];
    AllCKGradient=new Matrix***[NumOfC];
    CB=new double*[NumOfC];
    CBGradient=new double*[NumOfC];
    AllCBGradient=new double*[NumOfC];
    for(int i=0;i<NumOfC;i++)
    {
        CKernel[i]=new Matrix**[NumMapOfC[i]];
        CKGradient[i]=new Matrix**[NumMapOfC[i]];
        AllCKGradient[i]=new Matrix**[NumMapOfC[i]];

        CB[i]=new double[NumMapOfC[i]];
        CBGradient[i]=new double[NumMapOfC[i]];
        AllCBGradient[i]=new double[NumMapOfC[i]];
        for(int j=0;j<NumMapOfC[i];j++)
        {
            CB[i][j]=(((double)rand()/(RAND_MAX+1))-0.5)*2;
            if(i==0)
            {
                CKernel[i][j]=new Matrix*[1];
                CKGradient[i][j]=new Matrix*[1];
                AllCKGradient[i][j]=new Matrix*[1];
                CKGradient[i][j][0]=Matrix::CreateRand(Kernel,Kernel);
                AllCKGradient[i][j][0]=Matrix::CreateRand(Kernel,Kernel);
                CKernel[i][j][0]=Matrix::CreateRand(Kernel,Kernel);
            }
            else{
                CKernel[i][j]=new Matrix*[NumMapOfC[i-1]];
                CKGradient[i][j]=new Matrix*[NumMapOfC[i-1]];
                AllCKGradient[i][j]=new Matrix*[NumMapOfC[i-1]];
                for(int k=0;k<NumMapOfC[i-1];k++)
                {
                    CKernel[i][j][k]=Matrix::CreateRand(Kernel,Kernel);
                    CKGradient[i][j][k]=Matrix::CreateRand(Kernel,Kernel);
                    AllCKGradient[i][j][k]=Matrix::CreateRand(Kernel,Kernel);
                }
            }
        }
    }

    S=new Matrix**[NumOfS];
    SFeature=new Matrix**[NumOfS];
    SW=new double*[NumOfS];
    SWGradient=new double*[NumOfS];
    AllSWGradient=new double*[NumOfS];
    SB=new double*[NumOfS];
    SBGradient=new double*[NumOfS];
    AllSBGradient=new double*[NumOfS];

    for(int i=0;i<NumOfS;i++)
    {
        SW[i]=new double[NumMapOfC[i]];
        SWGradient[i]=new double[NumMapOfC[i]];
        AllSWGradient[i]=new double[NumMapOfC[i]];
        SB[i]=new double[NumMapOfC[i]];
        SBGradient[i]=new double[NumMapOfC[i]];
        AllSBGradient[i]=new double[NumMapOfC[i]];
        for(int j=0;j<NumMapOfC[i];j++)
        {
            SW[i][j]=(((double)rand()/(RAND_MAX+1))-0.5)*2;
            SB[i][j]=(((double)rand()/(RAND_MAX+1))-0.5)*2;
        }
    }

    int maps=inputmap;
    for(int i=0;i<NumOfC+NumOfS;i++)
    {
        if(i%2==0)
        {
            C[i/2]=new Matrix*[NumMapOfC[i/2]];
            CFeature[i/2]=new Matrix*[NumMapOfC[i/2]];
            for(int j=0;j<NumMapOfC[i/2];j++)
            {
                C[i/2][j]=new Matrix(maps-Kernel+1,maps-Kernel+1);
                CFeature[i/2][j]=new Matrix(maps-Kernel+1,maps-Kernel+1);
            }
            maps=maps-Kernel+1;
        }
        else{
            S[i/2]=new Matrix*[NumMapOfC[i/2]];
            SFeature[i/2]=new Matrix*[NumMapOfC[i/2]];
            for(int j=0;j<NumMapOfC[i/2];j++)
            {
                S[i/2][j]=new Matrix(maps/Scale,maps/Scale);
                SFeature[i/2][j]=new Matrix(maps/Scale,maps/Scale);
            }
            maps=maps/Scale;
        }
    }

    int x=maps*maps*NumMapOfC[NumOfC-1];
    Vec=new Matrix(x,1);
    F=new Matrix(x*2,1);
    WF=Matrix::CreateRand(x*2,x+1);
    WFGradient=new Matrix(x*2,x+1);
    AllWFGradient=new Matrix(x*2,x+1);
    Out=new Matrix(10,1);
    WOut=Matrix::CreateRand(10,x*2+1);
    WOutGradient=new Matrix(10,x*2+1);
    AllWOutGradient=new Matrix(10,x*2+1);
}

void CNN::FeedForward(Matrix &a)
{
    for(int i=0;i<NumMapOfC[0];i++)
    {
        C[0][i]=Matrix::Conv(C[0][i],a,*CKernel[0][i][0]);
        C[0][i]=Matrix::Add(C[0][i],CB[0][i],*C[0][i]);
    }

    for(int i=1;i<NumOfC+NumOfS;i++)
    {
        if(i%2==0)
        {
            Matrix **Midx=new Matrix*[NumMapOfC[i/2-1]];
            for(int j=0;j<NumMapOfC[i/2-1];j++)
                Midx[j]=new Matrix(C[i/2][0]->r,C[i/2][0]->c);
            for(int j=0;j<NumMapOfC[i/2];j++)
            {
                for(int k=0;k<NumMapOfC[i/2-1];k++)
                    Midx[k]=Matrix::Conv(Midx[k],*S[i/2-1][k],*CKernel[i/2][j][k]);
                C[i/2][j]=Matrix::AddMulti(C[i/2][j],NumMapOfC[i/2-1],Midx);
                C[i/2][j]=Matrix::Add(C[i/2][j],CB[i/2][j],*C[i/2][j]);
            }
            for(int k=0;k<NumMapOfC[i/2-1];k++)
                Midx[k]->~Matrix();
            delete[] Midx;

        }
        else{
            for(int j=0;j<NumMapOfC[i/2];j++)
            {
                S[i/2][j]=Matrix::Pooling(S[i/2][j],Scale,*C[i/2][j]);
                S[i/2][j]=Matrix::Time(S[i/2][j],SW[i/2][j],*S[i/2][j]);
                S[i/2][j]=Matrix::Add(S[i/2][j],SB[i/2][j],*S[i/2][j]);
               // S[i/2][j]->Show();
               // cout<<endl<<S[i/2][j]->r<<endl;
                S[i/2][j]=Matrix::Sigm(S[i/2][j],*S[i/2][j]);
               // S[i/2][j]->Show();
               // cout<<endl<<S[i/2][j]->r<<endl;
            }
        }
    }
    if(NumOfC>NumOfS)
    {
        Vec=Matrix::TMToVector(Vec,NumMapOfC[NumOfC-1],C[NumOfC-1]);
    }
    else{
        Vec=Matrix::TMToVector(Vec,NumMapOfC[NumOfC-1],S[NumOfC-1]);
    }
    Matrix *One=Matrix::CreateOnes(1,1);
    Matrix *VecAddOne=nullptr;
    VecAddOne=Matrix::CombineY(VecAddOne,*One,*Vec);

    F=Matrix::Time(F,*WF,*VecAddOne);
    F=Matrix::Sigm(F,*F);
    Matrix *FAddOne=nullptr;
    FAddOne=Matrix::CombineY(FAddOne,*One,*F);

    Out=Matrix::Time(Out,*WOut,*FAddOne);
    Out=Matrix::Sigm(Out,*Out);
    One->~Matrix();
    VecAddOne->~Matrix();
    FAddOne->~Matrix();
}

double CNN::GetError(Matrix &Y)
{
    Matrix *OneY=nullptr;
    OneY=Matrix::Minus(OneY,1,Y);
    Matrix *OneOut=nullptr;
    OneOut=Matrix::Minus(OneOut,1,*Out);
    Matrix *LogOneOut=nullptr;
    LogOneOut=Matrix::Log(LogOneOut,*OneOut);
    Matrix *LogOut=nullptr;
    LogOut=Matrix::Log(LogOut,*Out);
    Matrix *TimeOutY=nullptr;
    TimeOutY=Matrix::DotTime(TimeOutY,Y,*LogOut);
    Matrix *TimeOneOutY=nullptr;
    TimeOneOutY=Matrix::DotTime(TimeOneOutY,*OneY,*LogOneOut);
    Matrix *SumEr=nullptr;
    SumEr=Matrix::Add(SumEr,*TimeOutY,*TimeOneOutY);
    Matrix *SumErAll=nullptr;
    SumErAll=Matrix::SumAll(SumErAll,*SumEr);
    double ans=SumErAll->matrix[0][0]*(-1.0);
    OneY->~Matrix();
    OneOut->~Matrix();
    LogOut->~Matrix();
    LogOneOut->~Matrix();
    TimeOutY->~Matrix();
    TimeOneOutY->~Matrix();
    SumEr->~Matrix();
    SumErAll->~Matrix();
    return ans;
}

void CNN::ComputeGradient(Matrix &x,Matrix &y)
{
    Matrix *Mns=nullptr;
    Mns=Matrix::Minus(Mns,*Out,y);
    Matrix *MnsMulti=nullptr;
    MnsMulti=Matrix::CombineXMulti(MnsMulti,F->r,*Mns);
    Matrix *Ft=nullptr;
    Ft=Matrix::Transpose(Ft,*F);
    Matrix *ThetaOut=nullptr;
    ThetaOut=Matrix::DoubleTime(ThetaOut,Ft->matrix[0],*MnsMulti);
    WOutGradient=Matrix::CombineX(WOutGradient,*Mns,*ThetaOut);

    Matrix *ThetaF=nullptr;
    ThetaF=Matrix::DotTime(ThetaF,*WOut,*MnsMulti,1);
    Matrix *SumThetaF=nullptr;
    SumThetaF=Matrix::Sum(SumThetaF,*ThetaF);
    Matrix *SumThetaFt=nullptr;
    SumThetaFt=Matrix::Transpose(SumThetaFt,*SumThetaF);
    Matrix *OneMnsF=nullptr;
    OneMnsF=Matrix::Minus(OneMnsF,1,*F);
    SumThetaFt=Matrix::DotTime(SumThetaFt,*SumThetaFt,*F);
    SumThetaFt=Matrix::DotTime(SumThetaFt,*SumThetaFt,*OneMnsF);
    OneMnsF->~Matrix();
    //
    Matrix *SumThetaFtMulti=nullptr;
    SumThetaFtMulti=Matrix::CombineXMulti(SumThetaFtMulti,Vec->r,*SumThetaFt);
    Matrix *Vect=nullptr;
    Vect=Matrix::Transpose(Vect,*Vec);
    Matrix *ThetaWF=nullptr;
    ThetaWF=Matrix::DoubleTime(ThetaWF,Vect->matrix[0],*SumThetaFtMulti);
    WFGradient=Matrix::CombineX(WFGradient,*SumThetaFt,*ThetaWF);

    Matrix *ThetaVec=nullptr;
    ThetaVec=Matrix::DotTime(ThetaVec,*WF,*SumThetaFtMulti,1);
  //  cout<<WF->c<<endl;
  //  cout<<SumThetaFtMulti->c<<endl;
    Matrix *SumThetaVec=nullptr;
    SumThetaVec=Matrix::Sum(SumThetaVec,*ThetaVec);
    Matrix *SumThetaVect=nullptr;
    SumThetaVect=Matrix::Transpose(SumThetaVect,*SumThetaVec);
    Matrix **ThetaMap=nullptr;
    if(NumOfC>NumOfS)
    {
        CFeature[NumOfC-1]=Matrix::VectorToMatrix(CFeature[NumOfC-1],NumMapOfC[NumOfC-1],C[NumOfC-1][0]->r,C[NumOfC-1][0]->c,*SumThetaVect);
        Matrix *Sum=nullptr;
        for(int i=0;i<NumMapOfC[NumOfC-1];i++)
        {
            Sum=Matrix::SumAll(Sum,*CFeature[NumOfC-1][i]);
            CBGradient[NumOfC][i]=Sum->matrix[0][0];
            for(int j=0;j<NumMapOfC[NumOfC-2];j++)
            {
                CKGradient[NumOfC-1][i][j]=Matrix::Conv(CKGradient[NumOfC-1][i][j],*S[NumOfS-1][j],*CFeature[NumOfC-1][i]);
            }
        }
        Sum->~Matrix();
    }
    else
    {
        ThetaMap=Matrix::VectorToMatrix(ThetaMap,NumMapOfC[NumOfC-1],S[NumOfS-1][0]->r,S[NumOfS-1][0]->c,*SumThetaVect);
        Matrix * OneMnsS=nullptr;
        Matrix * Sum=nullptr;
        Matrix * PreMap=nullptr;
        Matrix * SumSWMap=nullptr;
        for(int i=0;i<NumMapOfC[NumOfS-1];i++)
        {
            OneMnsS=Matrix::Minus(OneMnsS,1,*S[NumOfS-1][i]);
            SFeature[NumOfS-1][i]=Matrix::DotTime(SFeature[NumOfS-1][i],*OneMnsS,*S[NumOfS-1][i]);
            SFeature[NumOfS-1][i]=Matrix::DotTime(SFeature[NumOfS-1][i],*SFeature[NumOfS-1][i],*ThetaMap[i]);
            PreMap=Matrix::Pooling(PreMap,2,*C[NumOfC-1][i]);
            SumSWMap=Matrix::DotTime(SumSWMap,*PreMap,*SFeature[NumOfS-1][i]);
            Sum=Matrix::SumAll(Sum,*SumSWMap);
            SWGradient[NumOfS-1][i]=Sum->matrix[0][0];
            Sum=Matrix::SumAll(Sum,*SFeature[NumOfS-1][i]);
            //cout<<SFeature[NumOfS-1][i]->matrix[0][0]<<endl;
            SBGradient[NumOfS-1][i]=Sum->matrix[0][0];
            ThetaMap[i]->~Matrix();
        }
        OneMnsS->~Matrix();
        Sum->~Matrix();
        PreMap->~Matrix();
        SumSWMap->~Matrix();
    }

    Mns->~Matrix();
    MnsMulti->~Matrix();
    Ft->~Matrix();
    ThetaOut->~Matrix();
    ThetaF->~Matrix();
    SumThetaF->~Matrix();
    SumThetaFt->~Matrix();
    SumThetaFtMulti->~Matrix();
    Vect->~Matrix();
    ThetaWF->~Matrix();
    ThetaVec->~Matrix();
    SumThetaVec->~Matrix();
    SumThetaVect->~Matrix();
    delete[] ThetaMap;

    for(int i=NumOfC+NumOfS-2;i>0;i--)
    {
        if(i%2==0)
        {
            Matrix *SFeatureTimeW=nullptr;
            Matrix *Sum=nullptr;
            for(int j=0;j<NumMapOfC[i/2];j++)
            {
                SFeatureTimeW=Matrix::Time(SFeatureTimeW,SW[i/2][j],*SFeature[i/2][j]);
                CFeature[i/2][j]=Matrix::CombineAll(CFeature[i/2][j],Scale,*SFeatureTimeW);
                for(int k=0;k<NumMapOfC[i/2-1];k++)
                {
                    CKGradient[i/2][j][k]=Matrix::Conv(CKGradient[i/2][j][k],*S[i/2-1][k],*CFeature[i/2][j]);
                }
                Sum=Matrix::SumAll(Sum,*CFeature[i/2][j]);
                CBGradient[i/2][j]=Sum->matrix[0][0];
            }
            SFeatureTimeW->~Matrix();
            Sum->~Matrix();
        }
        else{
            Matrix **AddSFeature;
            AddSFeature=new Matrix*[NumMapOfC[i/2+1]];
            for(int j=0;j<NumMapOfC[i/2];j++)
            {
                for(int k=0;k<NumMapOfC[i/2+1];k++)
                {
                    AddSFeature[k]=nullptr;
                    AddSFeature[k]=Matrix::ObConv(AddSFeature[k],*CFeature[i/2+1][k],*CKernel[i/2+1][k][j]);
                }
                SFeature[i/2][j]=Matrix::AddMulti(SFeature[i/2][j],NumMapOfC[i/2+1],AddSFeature);
                for(int k=0;k<NumMapOfC[i/2+1];k++)
                {
                    AddSFeature[k]->~Matrix();
                }
            }
            delete[] AddSFeature;
            Matrix * OneMnsS=nullptr;
            Matrix * Sum=nullptr;
            Matrix * PreMap=nullptr;
            Matrix * SumSWMap=nullptr;
            for(int j=0;j<NumMapOfC[i/2];j++)
            {
                OneMnsS=Matrix::Minus(OneMnsS,1,*S[i/2][j]);
                SFeature[i/2][j]=Matrix::DotTime(SFeature[i/2][j],*OneMnsS,*SFeature[i/2][j]);
                SFeature[i/2][j]=Matrix::DotTime(SFeature[i/2][j],*SFeature[i/2][j],*S[i/2][j]);
                PreMap=Matrix::Pooling(PreMap,2,*C[i/2][j]);
                SumSWMap=Matrix::DotTime(SumSWMap,*PreMap,*SFeature[i/2][j]);
                Sum=Matrix::SumAll(Sum,*SumSWMap);
                SWGradient[i/2][j]=Sum->matrix[0][0];
                Sum=Matrix::SumAll(Sum,*SFeature[i/2][j]);
                SBGradient[i/2][j]=Sum->matrix[0][0];
            }
            OneMnsS->~Matrix();
            PreMap->~Matrix();
            SumSWMap->~Matrix();
            Sum->~Matrix();
        }
    }
    Matrix *SFeatureTimeW=nullptr;
    Matrix *Sum=nullptr;
    for(int j=0;j<NumMapOfC[0];j++)
    {
        SFeatureTimeW=Matrix::Time(SFeatureTimeW,SW[0][j],*SFeature[0][j]);
        CFeature[0][j]=Matrix::CombineAll(CFeature[0][j],Scale,*SFeatureTimeW);
        CKGradient[0][j][0]=Matrix::Conv(CKGradient[0][j][0],x,*CFeature[0][j]);
        Sum=Matrix::SumAll(Sum,*CFeature[0][j]);
        CBGradient[0][j]=Sum->matrix[0][0];
    }
    SFeatureTimeW->~Matrix();
    Sum->~Matrix();
}


void CNN::ApplyGradient()
{
    for(int i=0;i<NumOfC;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            if(i==0)
            {
                AllCKGradient[0][j][0]=Matrix::Add(AllCKGradient[0][j][0],*AllCKGradient[0][j][0],*CKGradient[0][j][0]);
            }
            else{
                for(int k=0;k<NumMapOfC[i-1];k++)
                {
                    AllCKGradient[i][j][k]=Matrix::Add(AllCKGradient[i][j][k],*AllCKGradient[i][j][k],*CKGradient[i][j][k]);
                }
            }
            AllCBGradient[i][j]+=CBGradient[i][j];
        }
    }
    for(int i=0;i<NumOfS;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            AllSWGradient[i][j]+=SWGradient[i][j];
            AllSBGradient[i][j]+=SBGradient[i][j];
        }
    }
    AllWFGradient=Matrix::Add(AllWFGradient,*AllWFGradient,*WFGradient);
    AllWOutGradient=Matrix::Add(AllWOutGradient,*AllWOutGradient,*WOutGradient);
}

void CNN::SetZeros()
{
    for(int i=0;i<NumMapOfC[0];i++)
    {
        AllCKGradient[0][i][0]->SetZeros();
        AllCBGradient[0][i]=0;
    }
    for(int i=1;i<NumOfC;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            for(int k=0;k<NumMapOfC[i-1];k++)
            {
                AllCKGradient[i][j][k]->SetZeros();
            }
            AllCBGradient[i][j]=0;
        }
    }
    for(int i=0;i<NumOfS;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            AllSWGradient[i][j]=0;
            AllSBGradient[i][j]=0;
        }
    }
    AllWFGradient->SetZeros();
    AllWOutGradient->SetZeros();
}

void CNN::DGradient(int m,double Efficiency)
{
    for(int i=0;i<NumMapOfC[0];i++)
    {
        AllCKGradient[0][i][0]=Matrix::Time(AllCKGradient[0][i][0],Efficiency/m,*AllCKGradient[0][i][0]);
        CKernel[0][i][0]=Matrix::Minus(CKernel[0][i][0],*CKernel[0][i][0],*AllCKGradient[0][i][0]);
        CB[0][i]-=Efficiency/m*AllCBGradient[0][i];
    }
    for(int i=1;i<NumOfC;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            for(int k=0;k<NumMapOfC[i-1];k++)
            {
                AllCKGradient[i][j][k]=Matrix::Time(AllCKGradient[i][j][k],Efficiency/m,*AllCKGradient[i][j][k]);
                CKernel[i][j][k]=Matrix::Minus(CKernel[i][j][k],*CKernel[i][j][k],*AllCKGradient[i][j][k]);
            }
            CB[i][j]-=Efficiency/m*AllCBGradient[i][j];
        }
    }
    for(int i=0;i<NumOfS;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            SW[i][j]-=Efficiency/m*AllSWGradient[i][j];
            SB[i][j]-=Efficiency/m*AllSBGradient[i][j];
        }
    }
    AllWFGradient=Matrix::Time(AllWFGradient,Efficiency/m,*AllWFGradient);
    WF=Matrix::Minus(WF,*WF,*AllWFGradient);
    AllWOutGradient=Matrix::Time(AllWOutGradient,Efficiency/m,*AllWOutGradient);
    WOut=Matrix::Minus(WOut,*WOut,*AllWOutGradient);
}

void CNN::Train(Matrix **x,int num,int n,int batchsize,double Efficiency,Matrix **y)
{
    for(int i=0;i<num;i++)
    {
        for(int k=0;k<n/batchsize;k++)
        {
            SetZeros();
            double Error=0;
            for(int j=k*batchsize;j<(k+1)*batchsize;j++)
            {
                FeedForward(*x[j]);
                ComputeGradient(*x[j],*y[j]);
                ApplyGradient();
                Error+=GetError(*y[j]);
            }
            DGradient(batchsize,Efficiency);
            Error/=batchsize;
            cout<<Error<<endl;
        }
    }

}

void CNN::GetExample(Matrix ***x,Matrix ***y,int n)
{
    *x=new Matrix*[n];
    *y=new Matrix*[n];
    double ***X;
    X=new double**[n];
    double ***Y;
    Y=new double**[n];
    for(int i=0;i<n;i++)
    {
        X[i]=new double*[28];
        Y[i]=new double*[10];
        for(int j=0;j<28;j++)
            X[i][j]=new double[28];
        for(int j=0;j<10;j++)
        {
            Y[i][j]=new double[1];
            Y[i][j][0]=0;
        }
    }
    FILE *fp;
    fp=fopen("train-images.idx3-ubyte","rb");
    fseek(fp,16,SEEK_CUR);
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<28;j++)
        {
            for(int k=0;k<28;k++)
            {
                BYTE g;
                fscanf(fp,"%c",&g);
                X[i][j][k]=(double)g;
            }
        }
    }
    fclose(fp);
    fp=fopen("train-labels.idx1-ubyte","rb");
    fseek(fp,8,SEEK_CUR);
    for(int i=0;i<n;i++)
    {
        BYTE g;
        fscanf(fp,"%c",&g);
        int a=(int)g;
        Y[i][a][0]=1;
    }
    fclose(fp);
    for(int i=0;i<n;i++)
    {
        (*x)[i]=Matrix::CreateFromDouble(X[i],28,28);
        (*x)[i]=Matrix::Time((*x)[i],1.0/255,*(*x)[i]);
    //      x[i]->Show();
    //       cout<<endl;
        (*y)[i]=Matrix::CreateFromDouble(Y[i],10,1);
    //    y[i]->Show();
    //    cout<<endl;
    }
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<28;j++)
        {
            delete[] X[i][j];
        }
        delete[] X[i];
        for(int j=0;j<10;j++)
        {
            delete[] Y[i][j];
        }
        delete[] Y[i];
    }
    delete[] X;
    delete[] Y;
}

void CNN::GetExample1(Matrix ***x,Matrix ***y,int n)
{
    *x=new Matrix*[n];
    *y=new Matrix*[n];
    double ***X;
    X=new double**[n];
    double ***Y;
    Y=new double**[n];
    for(int i=0;i<n;i++)
    {
        X[i]=new double*[28];
        Y[i]=new double*[10];
        for(int j=0;j<28;j++)
            X[i][j]=new double[28];
        for(int j=0;j<10;j++)
        {
            Y[i][j]=new double[1];
            Y[i][j][0]=0;
        }
    }
    FILE *fp;
    fp=fopen("t10k-images.idx3-ubyte","rb");
    fseek(fp,16,SEEK_CUR);
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<28;j++)
        {
            for(int k=0;k<28;k++)
            {
                BYTE g;
                fscanf(fp,"%c",&g);
                X[i][j][k]=(double)g;
            }
        }
    }
    fclose(fp);
    fp=fopen("t10k-labels.idx1-ubyte","rb");
    fseek(fp,8,SEEK_CUR);
    for(int i=0;i<n;i++)
    {
        BYTE g;
        fscanf(fp,"%c",&g);
        int a=(int)g;
        Y[i][a][0]=1;
    }
    fclose(fp);
    for(int i=0;i<n;i++)
    {
        (*x)[i]=Matrix::CreateFromDouble(X[i],28,28);
        (*x)[i]=Matrix::Time((*x)[i],1.0/255,*(*x)[i]);
    //      x[i]->Show();
    //       cout<<endl;
        (*y)[i]=Matrix::CreateFromDouble(Y[i],10,1);
    //    y[i]->Show();
    //    cout<<endl;
    }
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<28;j++)
        {
            delete[] X[i][j];
        }
        delete[] X[i];
        for(int j=0;j<10;j++)
        {
            delete[] Y[i][j];
        }
        delete[] Y[i];
    }
    delete[] X;
    delete[] Y;
}

void CNN::Test(Matrix **x,Matrix **y,int n)
{
    double ans=0;
    for(int i=0;i<n;i++)
    {
        FeedForward(*x[i]);
        if(Out->GetMaxIndexVec()==y[i]->GetMaxIndexVec())
            ans++;
    }
    cout<<ans/n<<endl;
}

void CNN::WriteNetInFile()
{
    char str[50];
    cout<<"请输入文件名"<<endl;
    cin>>str;
    ofstream outfile(str,ios::binary);
    if(!outfile)
    {
        cout<<"cannot open the file"<<endl;
        exit(1);
    }
    outfile.write((char *)&inputmap,sizeof(int));
    outfile.write((char *)&NumOfC,sizeof(int));
    outfile.write((char *)&Kernel,sizeof(int));
    outfile.write((char *)&NumOfS,sizeof(int));
    outfile.write((char *)&Scale,sizeof(int));
    for(int i=0;i<NumOfC;i++)
        outfile.write((char *)&NumMapOfC[i],sizeof(int));
    for(int i=0;i<NumMapOfC[0];i++)
    {
        for(int j=0;j<Kernel;j++)
        {
            for(int k=0;k<Kernel;k++)
            {
                outfile.write((char *)&CKernel[0][i][0]->matrix[j][k],sizeof(double));
            }
        }
    }
    for(int i=1;i<NumOfC;i++)
    {
        for(int loop1=0;loop1<NumMapOfC[i];loop1++)
        {
            for(int loop2=0;loop2<NumMapOfC[i-1];loop2++)
            {
                for(int j=0;j<Kernel;j++)
                {
                    for(int k=0;k<Kernel;k++)
                    {
                        outfile.write((char *)&CKernel[i][loop1][loop2]->matrix[j][k],sizeof(double));
                    }
                }
            }
        }
    }
    for(int i=0;i<NumOfC;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            outfile.write((char *)&CB[i][j],sizeof(double));
        }
    }
    for(int i=0;i<NumOfS;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            outfile.write((char *)&SW[i][j],sizeof(double));
            outfile.write((char *)&SB[i][j],sizeof(double));
        }
    }
    for(int i=0;i<WF->r;i++)
    {
        for(int j=0;j<WF->c;j++)
        {
            outfile.write((char *)&WF->matrix[i][j],sizeof(double));
        }
    }
    for(int i=0;i<WOut->r;i++)
    {
        for(int j=0;j<WOut->c;j++)
        {
            outfile.write((char *)&WOut->matrix[i][j],sizeof(double));
        }
    }
    outfile.close();
}

void CNN::CheckGradient(Matrix *x,Matrix *y)
{
    double Epsilon=0.0001;
    for(int i=0;i<NumMapOfC[0];i++)
    {
        for(int k=0;k<5;k++){
                for(int j=0;j<5;j++){
                FeedForward(*x);
                ComputeGradient(*x,*y);

                cout<<CKGradient[0][i][0]->matrix[j][k]<<endl;

                CKernel[0][i][0]->matrix[j][k]+=Epsilon;
                FeedForward(*x);
                double sum1=GetError(*y);
                cout<<sum1<<endl;

                CKernel[0][i][0]->matrix[j][k]-=2*Epsilon;
                FeedForward(*x);
                double sum2=GetError(*y);
                cout<<sum2<<endl;
                cout<<(sum1-sum2)/(2*Epsilon)<<endl;

                CKernel[0][i][0]->matrix[j][k]+=Epsilon;

                getchar();
                getchar();}}

    }
}

CNN * CNN::GetNetFromFile()
{
    char str[50];
    cout<<"输入打开文件"<<endl;
    cin>>str;
    ifstream infile(str,ios::binary);
    if(!infile)
    {
        cout<<"cannot open the file"<<endl;
        exit(1);
    }
    int inputmap;
    int NumOfC;
    int Kernel;
    int NumOfS;
    int Scale;
    int *NumMapOfC=nullptr;
    infile.read((char *)&inputmap,sizeof(int));
    infile.read((char *)&NumOfC,sizeof(int));
    infile.read((char *)&Kernel,sizeof(int));
    infile.read((char *)&NumOfS,sizeof(int));
    infile.read((char *)&Scale,sizeof(int));
    NumMapOfC=new int[NumOfC];
    for(int i=0;i<NumOfC;i++)
        infile.read((char *)&NumMapOfC[i],sizeof(int));

    CNN*net=new CNN(inputmap,NumOfC,NumOfS,Kernel,Scale,NumMapOfC);
    for(int i=0;i<NumMapOfC[0];i++)
    {
        for(int j=0;j<Kernel;j++)
        {
            for(int k=0;k<Kernel;k++)
            {
                infile.read((char *)&net->CKernel[0][i][0]->matrix[j][k],sizeof(double));
            }
        }
    }
    for(int i=1;i<NumOfC;i++)
    {
        for(int loop1=0;loop1<NumMapOfC[i];loop1++)
        {
            for(int loop2=0;loop2<NumMapOfC[i-1];loop2++)
            {
                for(int j=0;j<Kernel;j++)
                {
                    for(int k=0;k<Kernel;k++)
                    {
                        infile.read((char *)&net->CKernel[i][loop1][loop2]->matrix[j][k],sizeof(double));
                    }
                }
            }
        }
    }
    for(int i=0;i<NumOfC;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            infile.read((char *)&net->CB[i][j],sizeof(double));
        }
    }
    for(int i=0;i<NumOfS;i++)
    {
        for(int j=0;j<NumMapOfC[i];j++)
        {
            infile.read((char *)&net->SW[i][j],sizeof(double));
            infile.read((char *)&net->SB[i][j],sizeof(double));
        }
    }
    for(int i=0;i<net->WF->r;i++)
    {
        for(int j=0;j<net->WF->c;j++)
        {
            infile.read((char *)&net->WF->matrix[i][j],sizeof(double));
        }
    }
    for(int i=0;i<net->WOut->r;i++)
    {
        for(int j=0;j<net->WOut->c;j++)
        {
            infile.read((char *)&net->WOut->matrix[i][j],sizeof(double));
        }
    }
    infile.close();
    return net;
}
