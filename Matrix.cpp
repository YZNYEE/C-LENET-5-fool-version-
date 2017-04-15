#include<iostream>
#include"Matrix.h"
#include"stdlib.h"
#include"time.h"
#include<cmath>
using namespace std;

Matrix::Matrix(int rr,int cc):r(rr),c(cc)
{
    matrix=new double*[r];
    for(int i=0;i<rr;i++)
        matrix[i]=new double[c];
}

Matrix::~Matrix()
{
    for(int i=0;i<r;i++)
        delete[] matrix[i];
    delete[] matrix;
}

Matrix * Matrix::CreateOnes(int rr,int cc)
{
    Matrix *m=new Matrix(rr,cc);
    for(int i=0;i<rr;i++)
    {
        for(int j=0;j<cc;j++)
            m->matrix[i][j]=1;
    }
    return m;
}

Matrix * Matrix::CreateZeros(int rr,int cc)
{
    Matrix *m=new Matrix(rr,cc);
    for(int i=0;i<rr;i++)
    {
        for(int j=0;j<cc;j++)
            m->matrix[i][j]=0;
    }
    return m;
}

Matrix * Matrix::Add(Matrix *m,Matrix &a,Matrix &b)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
            m->matrix[i][j]=a.matrix[i][j]+b.matrix[i][j];
    }
    return m;
}

Matrix * Matrix::Add(Matrix *m,double x,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
            m->matrix[i][j]=x+a.matrix[i][j];
    }
    return m;
}

Matrix * Matrix::Minus(Matrix * m,Matrix & a,Matrix & b)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            m->matrix[i][j]=a.matrix[i][j]-b.matrix[i][j];
        }
    }
    return m;
}

Matrix * Matrix::Minus(Matrix *m,double x,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            m->matrix[i][j]=x-a.matrix[i][j];
        }
    }
    return m;
}

Matrix * Matrix::Transpose(Matrix * m, Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.c,a.r);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
            m->matrix[i][j]=a.matrix[j][i];
    }
    return m;
}

Matrix * Matrix::DotTime(Matrix *m,Matrix &a,Matrix &b)
{
    if(m==nullptr)
        m=new Matrix(b.r,b.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
            m->matrix[i][j]=a.matrix[i][j]*b.matrix[i][j];
    }
    return m;
}

Matrix * Matrix::DotTime(Matrix *m,Matrix &a,Matrix &b,int n)
{
    if(m==nullptr)
        m=new Matrix(b.r,b.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
            m->matrix[i][j]=a.matrix[i][j+n]*b.matrix[i][j];
    }
    return m;
}

Matrix * Matrix::Time(Matrix *m, Matrix &a,Matrix &b)
{
    if(m==nullptr)
        m=new Matrix(a.r,b.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            double ans=0;
            for(int k=0;k<a.c;k++)
                ans+=a.matrix[i][k]*b.matrix[k][j];
            m->matrix[i][j]=ans;
        }
    }
    return m;
}

Matrix * Matrix::Time(Matrix *m,double x,Matrix & a)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
            m->matrix[i][j]=x*a.matrix[i][j];
    }
    return m;
}

Matrix * Matrix::DoubleTime(Matrix *m,double *x,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int j=0;j<m->c;j++)
    {
        for(int i=0;i<m->r;i++)
            m->matrix[i][j]=x[j]*a.matrix[i][j];
    }
    return m;
}

Matrix * Matrix::CombineX(Matrix *m,Matrix &a,Matrix &b)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c+b.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            if(j<a.c)
                m->matrix[i][j]=a.matrix[i][j];
            else
                m->matrix[i][j]=b.matrix[i][j-a.c];
        }
    }
    return m;
}

Matrix * Matrix::CombineY(Matrix *m,Matrix &a,Matrix &b)
{
    if(m==nullptr)
        m=new Matrix(a.r+b.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            if(i<a.c)
                m->matrix[i][j]=a.matrix[i][j];
            else
                m->matrix[i][j]=b.matrix[i-a.r][j];
        }
    }
    return m;
}

void Matrix::Show()
{
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
            cout<<matrix[i][j]<<' ';
        cout<<endl;
    }
}

Matrix * Matrix::CreateRand(int rr,int cc)
{
    Matrix * m=new Matrix(rr,cc);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            m->matrix[i][j]=((double)rand()/(RAND_MAX+1)-0.5)*2;
        }
    }
    return m;
}

Matrix * Matrix::Sigm(Matrix *m,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            m->matrix[i][j]=1.0/(1.0+exp(-1.0*a.matrix[i][j]));
        }
    }
    return m;
}

Matrix * Matrix::Log(Matrix *m,Matrix & a)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            m->matrix[i][j]=log(a.matrix[i][j]);
        }
    }
    return m;
}

Matrix * Matrix::Sum(Matrix *m,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(1,a.c);
    for(int j=0;j<m->c;j++)
    {
        double ans=0;
        for(int i=0;i<a.r;i++)
            ans+=a.matrix[i][j];
        m->matrix[0][j]=ans;
    }
    return m;
}

Matrix * Matrix::CreateFromDouble(double **a,int r,int c)
{
    Matrix *m=new Matrix(r,c);
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
            m->matrix[i][j]=a[i][j];
    }
    return m;
}

Matrix * Matrix::Conv(Matrix *m,Matrix &a,Matrix &k)
{
    if(m==nullptr)
        m=new Matrix(a.r-k.r+1,a.c-k.c+1);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            double ans=0;
            for(int loop1=0;loop1<k.r;loop1++)
            {
                for(int loop2=0;loop2<k.c;loop2++)
                {
                    ans+=a.matrix[i+loop1][j+loop2]*k.matrix[loop1][loop2];
                }
            }
            m->matrix[i][j]=ans;
        }
    }
    return m;
}

Matrix * Matrix::Pooling(Matrix *m,int s,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.r/s,a.c/s);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            double ans=0;
            for(int loop1=0;loop1<s;loop1++)
            {
                for(int loop2=0;loop2<s;loop2++)
                {
                    ans+=a.matrix[i*s+loop1][j*s+loop2];
                }
            }
            m->matrix[i][j]=ans;
        }
    }
    return m;
}

Matrix * Matrix::CombineXMulti(Matrix * m,int x,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.r,a.c*x);
    for(int k=0;k<x;k++)
    {
        for(int i=0;i<a.r;i++)
        {
            for(int j=0;j<a.c;j++)
            {
                m->matrix[i][k*a.c+j]=a.matrix[i][j];
            }
        }
    }
    return m;
}

Matrix * Matrix::ObConv(Matrix *m,Matrix &a,Matrix &k)
{
    if(m==nullptr)
        m=CreateZeros(a.r+k.r-1,a.c+k.c-1);
    else
        m->SetZeros();
    for(int i=0;i<a.r;i++)
    {
        for(int j=0;j<a.c;j++)
        {
            for(int loop1=0;loop1<k.r;loop1++)
            {
                for(int loop2=0;loop2<k.c;loop2++)
                {
                    m->matrix[i+loop1][j+loop2]+=a.matrix[i][j]*k.matrix[loop1][loop2];
                }
            }
        }
    }
    return m;
}

void Matrix::SetZeros()
{
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            matrix[i][j]=0;
        }
    }
}

Matrix * Matrix::SumAll(Matrix * m,Matrix & a)
{
    if(m==nullptr)
        m=new Matrix(1,1);
    double ans=0;
    for(int i=0;i<a.r;i++)
    {
        for(int j=0;j<a.c;j++)
        {
            ans+=a.matrix[i][j];
        }
    }
    m->matrix[0][0]=ans;
    return m;
}

Matrix * Matrix::AddMulti(Matrix * m,int x,Matrix **a)
{
    if(m==nullptr)
        m=new Matrix(a[0]->r,a[0]->c);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            double ans=0;
            for(int k=0;k<x;k++)
                ans+=a[k]->matrix[i][j];
            m->matrix[i][j]=ans;
        }
    }
    return m;
}

Matrix * Matrix::TMToVector(Matrix *m,int x,Matrix **a)
{
    int Size=a[0]->r*a[0]->c;
    if(m==nullptr)
        m=new Matrix(Size*x,1);
    for(int k=0;k<x;k++)
    {
        for(int i=0;i<a[0]->r;i++)
        {
            for(int j=0;j<a[0]->c;j++)
            {
                m->matrix[Size*k+i*a[0]->r+j][0]=a[k]->matrix[i][j];
            }
        }
    }
    return m;
}

Matrix ** Matrix::VectorToMatrix(Matrix **m,int x,int r,int c,Matrix &a)
{
    if(m==nullptr)
    {
        m=new Matrix*[x];
        for(int i=0;i<x;i++)
            m[i]=new Matrix(r,c);
    }
    int Index=0;
    for(int i=0;i<x;i++)
    {
        for(int loop1=0;loop1<r;loop1++)
        {
            for(int loop2=0;loop2<c;loop2++)
            {
                m[i]->matrix[loop1][loop2]=a.matrix[Index][0];
                Index++;
            }
        }
    }
    return m;
}

Matrix *Matrix::CombineAll(Matrix *m,int x,Matrix &a)
{
    if(m==nullptr)
        m=new Matrix(a.r*x,a.c*x);
    for(int i=0;i<m->r;i++)
    {
        for(int j=0;j<m->c;j++)
        {
            m->matrix[i][j]=a.matrix[i/x][j/x];
        }
    }
    return m;
}

int Matrix::GetMaxIndexVec()
{
    int index;
    double ans=0;
    for(int i=0;i<r;i++)
    {
        if(matrix[i][0]>ans)
        {
            index=i;
            ans=matrix[i][0];
        }
    }
    return index;
}
