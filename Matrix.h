#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

/*�����࣬BP��CNN����Ӧ�õ������������ɵ�����*/
class Matrix
{
public:
    Matrix(int r,int c);
    ~Matrix();
    //����ȫ1����r��c��ά��
    static Matrix * CreateOnes(int r,int c);
    //����0����
    static Matrix * CreateZeros(int r,int c);
    //����0-1�������
    static Matrix * CreateRand(int,int);
    //��double **����ת��Ϊ����,���������������
    static Matrix * CreateFromDouble(double **a,int r,int c);
    //�������
    static Matrix * Add(Matrix *,Matrix &,Matrix &);
    //������ʵ����ӣ���ÿ��Ԫ����ʵ��
    static Matrix * Add(Matrix *,double ,Matrix &);
    //���������ӷ���һ������
    static Matrix * AddMulti(Matrix *,int ,Matrix **);
    //�������
    static Matrix * Minus(Matrix *,Matrix &,Matrix &);
    //ʵ��������
    static Matrix * Minus(Matrix *,double x,Matrix &);
    //����ת��
    static Matrix * Transpose(Matrix *,Matrix &);
    //������
    static Matrix * DotTime(Matrix *,Matrix &,Matrix &);
    //ƫ�ȵ�ˣ��Ժ���ͣ���ȫΪ�����㷽��
    static Matrix * DotTime(Matrix *,Matrix &,Matrix &,int n);
    //���
    static Matrix * Time(Matrix *,Matrix &,Matrix &);
    //ʵ����������
    static Matrix * Time(Matrix *,double,Matrix &);
    //ʵ�������������ˣ��Ժ���ͣ����ֱ�ÿ��ʵ������Ӧ�����
    static Matrix * DoubleTime(Matrix *,double *,Matrix &);
    //���������������
    static Matrix * CombineX(Matrix *,Matrix &,Matrix &);
    //��������
    static Matrix * CombineY(Matrix *,Matrix &,Matrix &);
    //������չ���ڼ����ݶȣ�s�㷵�ص�c��ʱ�õ�
    static Matrix * CombineAll(Matrix *,int,Matrix &);
    //��������������
    static Matrix * CombineXMulti(Matrix *,int,Matrix &);
    //����sigmoid����
    static Matrix * Sigm(Matrix *,Matrix &);
    //����Log�������������ʱ�õ�
    static Matrix * Log(Matrix *,Matrix &);
    //����������ͣ�����һ��1*n�ľ���
    static Matrix * Sum(Matrix *,Matrix &);
    //��������Ԫ�����
    static Matrix * SumAll(Matrix *,Matrix &);
    //�������
    static Matrix * Conv(Matrix *,Matrix &,Matrix &);
    //�������� �ڷ��򴫵�ʱ����
    static Matrix * ObConv(Matrix *,Matrix &,Matrix &);
    //pooling����
    static Matrix * Pooling(Matrix *,int,Matrix &);
    //�������ת��Ϊ�����������һ�㵽BP�����õ�
    static Matrix * TMToVector(Matrix *,int,Matrix **);
    //����ת��Ϊ��������ڷ��򴫵�ʱ�õ�
    static Matrix ** VectorToMatrix(Matrix **,int,int,int,Matrix &);
    //��ȡn*1����������Ԫ��
    int GetMaxIndexVec();
    //��������
    void SetZeros();

    //�������޹ؽ�Ҫ�ĺ���
    int ReR() const
    {
        return r;
    };
    int ReC() const
    {
        return c;
    };
    double *GetR(int x)
    {
        return matrix[x];
    }
    double GetNum(int n,int m) const
    {
        return matrix[n][m];
    }
    void Show();
//private:
    //������ĳ�Ա
    double ** matrix;
    int r;
    int c;
};


#endif // MATRIX_H_INCLUDED
