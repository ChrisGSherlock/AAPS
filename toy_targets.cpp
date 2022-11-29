#include <iostream>
#include <Eigen/Core>
#include <random>

using namespace std;
using namespace Eigen;

// To test on its own, remove the comments around main() then
// g++ -I /usr/include/eigen3 toy_targets_Eigen.cpp
// When linking with another main file: #include "toy_targets_Eigen.h"

//----------------------------------

// Null functions
// For flat priors or flat likelihood
double l_null(const ArrayXd &x, const ArrayXd &thetas) {
  return 0.0;
}
VectorXd gl_null(const ArrayXd &x, const ArrayXd &thetas) {
  return x*0;
}
double gldote_null(const ArrayXd &x, const VectorXd &e, const ArrayXd &thetas) {
  return 0.0;
}

//**********ISOTROPIC****************

// Isotropic Gaussian
double l_GaussIso(const ArrayXd &x, const ArrayXd &thetas) {
  double norm=x.matrix().norm();
  double sig2=thetas[0];
  return -0.5*norm*norm/sig2;
}
VectorXd gl_GaussIso(const ArrayXd &x, const ArrayXd &thetas) {
  double sig2=thetas[0];
  return -x.matrix()/sig2;
}
double gldote_GaussIso(const ArrayXd &x, const VectorXd &e, const ArrayXd &thetas) {
  double sig2=thetas[0];
  return -e.dot(x.matrix())/sig2;
}

// Power isotropic log f = -||x||^a/a where ||x|| is the L2 norm
double l_PowIso(const ArrayXd &x, const ArrayXd &theta) {
  double norm=x.matrix().norm();
  double a=theta[0];
  return -exp(a*log(norm))/a;
}
VectorXd gl_PowIso(const ArrayXd &x, const ArrayXd &theta) {
  double norm=x.matrix().norm();
  double a=theta[0];
  return -exp((a-2)*log(norm))*x.matrix();
}
double gldote_PowIso(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  double norm=x.matrix().norm();
  double a=theta[0];
  return -exp((a-2)*log(norm))*e.dot(x.matrix());
}

//**********MORE GENERAL GAUSSIAN************

// Diagonal Gaussian specifying each expectation and diagonal variance.
// Expectation \ne 0 can be useful when using this as a prior.
double l_GaussDiag(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  ArrayXd E=theta.head(d), V=theta.tail(d);
  ArrayXd lterms=(x-E)*(x-E)/V;
  return -0.5*lterms.sum();
}
VectorXd gl_GaussDiag(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  ArrayXd E=theta.head(d), V=theta.tail(d);
  ArrayXd gterms=-(x-E)/V;
  return gterms.matrix();
}
double gldote_GaussDiag(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  return e.dot(gl_GaussDiag(x,theta));
}

// SDs linear from theta[0] to theta[1]
double l_GaussLinSD(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd sds=ArrayXd::LinSpaced(x.size(),theta[0],theta[1]);
  ArrayXd tmp=x*x/sds/sds;
  return -0.5*tmp.sum();
}
VectorXd gl_GaussLinSD(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd sds=ArrayXd::LinSpaced(x.size(),theta[0],theta[1]);
  ArrayXd tmp=-x/(sds*sds);
  return tmp.matrix();
}
double gldote_GaussLinSD(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  return e.dot(gl_GaussLinSD(x,theta));
}

// Diag(Hessian) linear from theta[0] to theta[1]
double l_GaussLinH(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd Hdiag=ArrayXd::LinSpaced(x.size(),theta[0],theta[1]);
  ArrayXd tmp=x*x*Hdiag;
  return -0.5*tmp.sum();
}
VectorXd gl_GaussLinH(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd Hdiag=ArrayXd::LinSpaced(x.size(),theta[0],theta[1]);
  ArrayXd tmp=-x*Hdiag;
  return tmp.matrix();
}
double gldote_GaussLinH(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  return e.dot(gl_GaussLinH(x,theta));
}


//*************** MORE GENERAL EXPONENTIAL POWER LAW **************

// Power aniso log f = -||x/scvec||^a/a theta=(a,lo,hi)
double l_PowLinSc(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double norm=xstd.matrix().norm();
  double a=theta[0];
  return -exp(a*log(norm))/a;
}
VectorXd gl_PowLinSc(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double norm=xstd.matrix().norm();
  double a=theta[0];
  ArrayXd g=-exp((a-2)*log(norm))*xstd/scs;
  return g.matrix();
}
double gldote_PowLinSc(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  return e.dot(gl_PowLinSc(x,theta));
}

// Product of independent powers: log f = -sum |x_i/sc_i|^a/a theta=(a,lo,hi)
double l_PowProdLinSc(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double a=theta[0];
  ArrayXd terms=exp(a*log(xstd.abs()));
  return -terms.sum()/a;
}
VectorXd gl_PowProdLinSc(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double am1=theta[0]-1;
  ArrayXd g=-exp(am1*log(xstd.abs()))*xstd.sign()/scs;
  return g.matrix();
}
double gldote_PowProdLinSc(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  return e.dot(gl_PowProdLinSc(x,theta));
}

//************** Logistic *******************
// Product of logistics along the co-ordinate axes
// vector specifies (E1,...,Ed,Sc1,...,Scd)
double l_LogisticCpts(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  ArrayXd E=theta.head(d), Sc=theta.tail(d);
  ArrayXd std=(x-E)/Sc; // standardised
  double m=std.maxCoeff(); // for robustness to large std
  return (std-2*(m+log(exp(-m)+exp(std-m)))-log(Sc)).sum();
}
VectorXd gl_LogisticCpts(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  ArrayXd E=theta.head(d), Sc=theta.tail(d);
  ArrayXd std=(x-E)/Sc;
  ArrayXd gterms=1/Sc * (1-2/(1+exp(-std)));
  return gterms.matrix();
}



//*************** Student-t ******************

// Diagonal multivariate Student-t with linear scalings
double l_MVtDiagLinSc(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  double df=theta(0);
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double denom=1+(xstd*xstd).sum()/df;
  return -(df+(double)d)/2*log(denom);
}
VectorXd gl_MVtDiagLinSc(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  double df=theta(0);
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double denom=1+(xstd*xstd).sum()/df;
  return -(df+(double)d)/df*xstd/scs/denom;
}
double gldote_MVtDiagLinSc(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  return e.dot(gl_MVtDiagLinSc(x,theta));
}

// Product of univariate Student-ts with linear scalings
double l_tProdLinSc(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double df=theta[0];
  ArrayXd terms=-(df+1)/2*log(1+xstd*xstd/df);
  return terms.sum();
}
VectorXd gl_tProdLinSc(const ArrayXd &x, const ArrayXd &theta) {
  ArrayXd scs=ArrayXd::LinSpaced(x.size(),theta[1],theta[2]);
  ArrayXd xstd=x/scs;
  double df=theta[0];
  ArrayXd terms=-(df+1)/df*xstd/scs/(1+xstd*xstd/df);
  return terms.matrix();
}
double gldote_tProdLinSc(const ArrayXd &x, const VectorXd &e, const ArrayXd &theta) {
  return e.dot(gl_tProdLinSc(x,theta));
}

// Multivariate Student-t with general inverse variance matrix
double l_MVt(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  double df=theta(0);
  ArrayXd Vinvlong=theta.tail(d*d);
  Map<MatrixXd> Vinv(Vinvlong.data(),d,d);
  VectorXd xv=x.matrix();
  double denom=1+(xv.transpose()*Vinv*xv)(0,0)/df;
  return  - (df+(double)d)/2.0 *log(denom);
}

// *********Skew Normal*********
// **Ancillary functions**
const double _PI = 3.14159265358979323846;
double Phi(const double x) {
  const double sqrt2=sqrt(2.0);
  return 0.5*(1.0+std::erf(x/sqrt2));
}
double logPhi(const double x) {
  if (x>-7) {
    return log(Phi(x));
  }
  else { // x is negative, remember!
    double x2=x*x, x3=x2*x, x5=x3*x*x, x7=x5*x*x;
    return -0.5*log(2*_PI)-0.5*x*x+log(-1/x+1/x3-3/x5+15/x7);
  }
}
double phiOPhi(const double x) {
  if (x>-7) {
    return 1/sqrt(2*_PI)*exp(-x*x/2)/Phi(x);
  }
  else {
    double x3=x*x*x, x5=x3*x*x, x7=x5*x*x;
    return 1/(-1/x+1/x3-3/x5+15/x7);
  }
}
ArrayXd Phi(const ArrayXd &xs) {
  ArrayXd Phis=xs;
  int i;
  const int n=xs.size();
  const double sqrt2=sqrt(2.0);
  
  for (i=0; i<n; i++) {
    Phis(i)=0.5*(1+std::erf(xs(i)/sqrt2));
  }
  return Phis;
}
ArrayXd logPhi(const ArrayXd &xs) {
  ArrayXd logPhis=xs;
  int i;
  const int n=xs.size();

  for (i=0; i<n; i++) {
    logPhis(i)=logPhi(xs(i));
  }
  return logPhis;
}
ArrayXd phiOPhi(const ArrayXd &xs) {
  ArrayXd rats=xs;
  int i;
  const int n=xs.size();

  for (i=0; i<n; i++) {
    rats(i)=phiOPhi(xs(i));
  }
  return rats;
}
// Product of skey normals along the co-ordinate axes
// vector specifies (alpha,Sc1,...,Scd)
double l_SkewNormalCpts(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  double alph=theta(0);
  ArrayXd Sc=theta.tail(d), std=x/Sc; // standardised
  return (log(2/Sc)-0.5*std*std-0.5*log(2*_PI)+logPhi(alph*std)).sum();
}
VectorXd gl_SkewNormalCpts(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  double alph=theta(0);
  ArrayXd Sc=theta.tail(d), std=x/Sc;
  ArrayXd gterms= -std/Sc + alph/Sc * phiOPhi(alph*std);
  return gterms.matrix();
}


//************** Modified Rosenbrock *******************
// Product of bananas along pairs of co-ordinate axes
// vector specifies (beta,Sc_1,...,Sc_{d/2})
static void CopyToHalfSize(const ArrayXd &full, ArrayXd &half, bool odd) {
  int nfull=full.size(), nhalf=half.size(), i;
  if (nhalf*2 != nfull) {
    cout <<"CopyHalfToNew size mismatch: 2x"<<nhalf<<"!="<<nfull<<endl;
    exit(1);
  }
  for (i=0;i<nhalf;i++) {
    half(i)=full(2*i+odd);
  }
}
static void CopyFromHalfSize(const ArrayXd &half, ArrayXd &full, bool odd) {
  int nfull=full.size(), nhalf=half.size(), i;
  if (nhalf*2 != nfull) {
    cout <<"CopyHalfToNew size mismatch: 2x"<<nhalf<<"!="<<nfull<<endl;
    exit(1);
  }
  for (i=0;i<nhalf;i++) {
    full(2*i+odd)=half(i);
  }
}

double l_ModifiedRosenbrock(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  double sq2=sqrt(2);
  if ((int) ( ((double)d)/2) != d/2) {
    cout <<"Modified Rosenbrock: dimension must be even!\n";
    exit(1);
  }
  ArrayXd Sc=theta.tail(d/2);
  double beta=theta(0);
  ArrayXd xA=ArrayXd::Zero(d/2), xB=ArrayXd::Zero(d/2);
  CopyToHalfSize(x,xA,false); CopyToHalfSize(x,xB,true);
  ArrayXd stdA=xA/Sc-sq2*beta;
  ArrayXd tmp=1+xA*xA/4.0/Sc/Sc;
  ArrayXd stdB=(xB-xA*xA/tmp/sq2/Sc);
  stdA=stdA*stdA; stdB=stdB*stdB;
  return -0.5*stdA.sum() - 0.5*stdB.sum();
}
VectorXd gl_ModifiedRosenbrock(const ArrayXd &x, const ArrayXd &theta) {
  int d=x.size();
  double sq2=sqrt(2);
  if ((int) ( ((double)d)/2) != d/2) {
    cout <<"Modified Rosenbrock: dimension must be even!\n";
    exit(1);
  }
  ArrayXd Sc=theta.tail(d/2);
  double beta=theta(0);
  ArrayXd xA=ArrayXd::Zero(d/2), xB=ArrayXd::Zero(d/2);
  CopyToHalfSize(x,xA,false); CopyToHalfSize(x,xB,true);
  ArrayXd stdA=xA/Sc-sq2*beta;
  ArrayXd tmp=1+xA*xA/4.0/Sc/Sc;
  ArrayXd stdB=(xB-xA*xA/tmp/sq2/Sc);
  ArrayXd dxB= - stdB;
  ArrayXd dxA= -stdA/Sc+stdB*sq2*xA/Sc/tmp/tmp;
  ArrayXd gterms=ArrayXd::Zero(d);
  CopyFromHalfSize(dxA,gterms,false); CopyFromHalfSize(dxB,gterms,true);

  return gterms.matrix();
}


// ******NUMERICAL DIFFERENTIATION*******
// Can be used to check the actual differentiation or to create bespoke
// gradient functions that use the numerical derivative

VectorXd glp_from_lp(const ArrayXd &xs, const ArrayXd &theta, double (*plp_fn)(const ArrayXd &xs, const ArrayXd &thetas), const double eps=1e-5) {
  int i, d=xs.size();
  VectorXd glp=VectorXd(d);
  ArrayXd ei=ArrayXd::Zero(d);
  for (i=0;i<d;i++) {
    ei(i)=eps;
    glp[i]=((*plp_fn)(xs+ei,theta)-(*plp_fn)(xs-ei,theta))/(2*eps);
    ei(i)=0;
  }
  return glp;
}

MatrixXd Hlp_from_lp(const ArrayXd &xs, const ArrayXd &theta, double (*plp_fn)(const ArrayXd &xs, const ArrayXd &theta), const double eps=1e-5) {
  int i,j, d=xs.size();
  MatrixXd Hlp=MatrixXd(d,d);
  ArrayXd ei=ArrayXd::Zero(d), ej=ArrayXd::Zero(d);
  for (i=0;i<d;i++) {
    ei(i)=eps;
    for (j=0;j<d;j++) {
      ej(j)=eps;
      Hlp(i,j)=(*plp_fn)(xs+ei+ej,theta)+(*plp_fn)(xs-ei-ej,theta)
	-(*plp_fn)(xs-ei+ej,theta)-(*plp_fn)(xs+ei-ej,theta);
      Hlp(i,j)=Hlp(i,j)/(4*eps*eps);
      ej(j)=0.0;
    }
    ei(i)=0.0;
  }
  return Hlp;
}


// ************* Random samples ***********************

Eigen::ArrayXd StdNormalAXd(size_t n, std::mt19937 &gen, std::normal_distribution<double> StdNormal) {
  size_t i;
  Eigen::ArrayXd out=Eigen::ArrayXd(n);
  for (i=0;i<n;i++) {
    out(i)=StdNormal(gen);
  }
  return out;
}

Eigen::ArrayXd StdSkewNormalAXd(size_t n, double omega, std::mt19937 &gen, std::normal_distribution<double> StdNormal) {
  size_t i;
  Eigen::ArrayXd out=Eigen::ArrayXd(n);
  for (i=0;i<n;i++) {
    double tmp1=StdNormal(gen), tmp2=StdNormal(gen);
    if (tmp2<omega*tmp1) { // has probability Phi(omega tmp1)
      out(i)=tmp1;
    }
    else {
      out(i)=-tmp1;
    }
  }
  return out;
}

Eigen::ArrayXd StdUnifAXd(size_t n, std::mt19937 &gen, std::uniform_real_distribution<double> StdUnif) {
  size_t i;
  Eigen::ArrayXd out=Eigen::ArrayXd(n);
  for (i=0;i<n;i++) {
    out(i)=StdUnif(gen);
  }
  return out;
}

Eigen::ArrayXd StdLogisticAXd(size_t n, std::mt19937 &gen, std::uniform_real_distribution<double> StdUnif) {
  size_t i;
  Eigen::ArrayXd out=Eigen::ArrayXd(n);
  for (i=0;i<n;i++) {
    out(i)=StdUnif(gen);
  }
  return log(out/(1-out));
}

Eigen::ArrayXd ModifiedRosenbrockAXd(size_t n, std::mt19937 &gen, std::normal_distribution<double> StdNormal, const ArrayXd &theta) {
  size_t i;
  Eigen::ArrayXd out=Eigen::ArrayXd(n);
  double sq2=sqrt(2);
  if ((int) ( ((double)n)/2) != n/2) {
    cout <<"Modified Rosenbrock: dimension must be even!\n";
    exit(1);
  }
  ArrayXd Sc=theta.tail(n/2);
  double beta=theta(0);

  for (i=0;i<n/2;i++) {
    out(2*i)=StdNormal(gen)*Sc(i)+sq2*beta*Sc(i);
    double tmp=out(2*i)*out(2*i)/4.0/Sc(i)/Sc(i);
    out(2*i+1)=StdNormal(gen)+4*Sc(i)/sq2*tmp/(1+tmp);
  }
  return out;
}



/*
// Test 
int main() {
  int d=4,i,j;
  ArrayXd x=ArrayXd::LinSpaced(d,1,d);
  double l,a;
  VectorXd gl;
  VectorXd e=VectorXd::Zero(d); e(0)=1;
  
  cout << "x="<<x.matrix().transpose()<<endl;

  cout << "Null functions\n";
  ArrayXd thetanull(1); thetanull << 1;
  l=l_null(x,thetanull);
  gl=gl_null(x,thetanull);
  a=gldote_null(x,e,thetanull);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;

  ArrayXd thetaGaussIso(1); thetaGaussIso <<2;
  cout << "Isotropic Gaussian with sigma2="<<thetaGaussIso[0]<<endl;
  l=l_GaussIso(x,thetaGaussIso);
  gl=gl_GaussIso(x,thetaGaussIso);
  a=gldote_GaussIso(x,e,thetaGaussIso);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetaGaussIso, &l_GaussIso);
  cout << gl.transpose()<<endl;

  ArrayXd thetaPowIso(1); thetaPowIso <<4; // is the power
  cout << "Isotropic L2 norm to power of "<<thetaPowIso[0]<<endl;
  l=l_PowIso(x,thetaPowIso);
  gl=gl_PowIso(x,thetaPowIso);
  a=gldote_PowIso(x,e,thetaPowIso);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetaPowIso, &l_PowIso);
  cout << gl.transpose()<<endl;

  ArrayXd thetaGaussDiag(2*d);
  thetaGaussDiag.head(d)=ArrayXd::LinSpaced(d,0,d-1);
  thetaGaussDiag.tail(d)=ArrayXd::LinSpaced(d,1,d);
  cout << "Gaussian expectations and diagonal variances: ";
  cout << thetaGaussDiag.matrix().transpose()<<endl;
  l=l_GaussDiag(x,thetaGaussDiag);
  gl=gl_GaussDiag(x,thetaGaussDiag);
  a=gldote_GaussDiag(x,e,thetaGaussDiag);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetaGaussDiag, &l_GaussDiag);
  cout << gl.transpose()<<endl;

  ArrayXd thetaGaussLinSD(2); thetaGaussLinSD <<1,10; // is the range
  cout << "Gaussian with SDs linear between "<<thetaGaussLinSD(0)<<" and "<<thetaGaussLinSD(1)<<endl;
  l=l_GaussLinSD(x,thetaGaussLinSD);
  gl=gl_GaussLinSD(x,thetaGaussLinSD);
  a=gldote_GaussLinSD(x,e,thetaGaussLinSD);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetaGaussLinSD, &l_GaussLinSD);
  cout << gl.transpose()<<endl;

  ArrayXd thetaGaussLinH(2); thetaGaussLinH <<1,10; // is the range
  cout << "Gaussian with diag(H) linear between "<<thetaGaussLinH(0)<<" and "<<thetaGaussLinH(1)<<endl;
  l=l_GaussLinH(x,thetaGaussLinH);
  gl=gl_GaussLinH(x,thetaGaussLinH);
  a=gldote_GaussLinH(x,e,thetaGaussLinH);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  cout << "Numerical differentiation.\n Gradient:";
  gl=glp_from_lp(x, thetaGaussLinH, &l_GaussLinH);
  cout << gl.transpose()<<endl<<"Hessian:"<<endl;
  MatrixXd H=Hlp_from_lp(x,thetaGaussLinH,&l_GaussLinH);
  cout <<H<<endl;

  ArrayXd thetaPowLinSc(3); thetaPowLinSc <<4,1,10; // (pow,lo,hi)
  cout << "Exponential power "<<thetaPowLinSc(0)<<" with scales linear between "<<thetaPowLinSc(1)<<" and "<<thetaPowLinSc(2)<<endl;
  l=l_PowLinSc(x,thetaPowLinSc);
  gl=gl_PowLinSc(x,thetaPowLinSc);
  a=gldote_PowLinSc(x,e,thetaPowLinSc);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetaPowLinSc, &l_PowLinSc);
  cout << gl.transpose()<<endl;

  ArrayXd thetaPowProdLinSc(3); thetaPowProdLinSc <<4,1,10; // (pow,lo,hi)
  cout << "Exponential power "<<thetaPowProdLinSc(0)<<" product, with scales linear between "<<thetaPowProdLinSc(1)<<" and "<<thetaPowProdLinSc(2)<<endl;
  l=l_PowProdLinSc(x,thetaPowProdLinSc);
  gl=gl_PowProdLinSc(x,thetaPowProdLinSc);
  a=gldote_PowProdLinSc(x,e,thetaPowProdLinSc);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetaPowProdLinSc, &l_PowProdLinSc);
  cout << gl.transpose()<<endl;

  ArrayXd thetaLogis(2*d);
  thetaLogis.head(d)=ArrayXd::LinSpaced(d,0,0); // Expectations
  thetaLogis.tail(d)=ArrayXd::LinSpaced(d,1,10); // Scale parameters
  cout << "Product of Logistics with [E1,..,,Ed,Sc1,...,Scd]=\n"<< thetaLogis.matrix().transpose()<<endl;
  l=l_LogisticCpts(x,thetaLogis);
  gl=gl_LogisticCpts(x,thetaLogis);
  cout << l <<endl<<gl.transpose()<<endl;
  gl=glp_from_lp(x, thetaLogis, &l_LogisticCpts);
  cout << gl.transpose()<<endl;


  ArrayXd thetaMVtDiagLinSc(3); thetaMVtDiagLinSc <<10,1,10; // is the range
  cout << "MV Student-t with df="<<thetaMVtDiagLinSc(0)<<" and scalings linear between "<<thetaMVtDiagLinSc(1)<<" and "<<thetaMVtDiagLinSc(2)<<endl;
  l=l_MVtDiagLinSc(x,thetaMVtDiagLinSc);
  gl=gl_MVtDiagLinSc(x,thetaMVtDiagLinSc);
  a=gldote_MVtDiagLinSc(x,e,thetaMVtDiagLinSc);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetaMVtDiagLinSc, &l_MVtDiagLinSc);
  cout << gl.transpose()<<endl;
  
  ArrayXd thetatProdLinSc(3); thetatProdLinSc <<10,1,10; // is the range
  cout << "Product of Student-t with df="<<thetatProdLinSc(0)<<" and scalings linear between "<<thetatProdLinSc(1)<<" and "<<thetatProdLinSc(2)<<endl;
  l=l_tProdLinSc(x,thetatProdLinSc);
  gl=gl_tProdLinSc(x,thetatProdLinSc);
  a=gldote_tProdLinSc(x,e,thetatProdLinSc);
  cout << l <<endl<<gl.transpose()<<endl<<a<<endl;
  gl=glp_from_lp(x, thetatProdLinSc, &l_tProdLinSc);
  cout << gl.transpose()<<endl;

  ArrayXd thetaMVt(1+d*d);
  for (i=0;i<d;i++) { for (j=0;j<d;j++) {
      thetaMVt(0)=10; // df
      thetaMVt(1+i+d*j)=exp((double)abs(i-j)*log(0.5)); // inverse variance
    }}
  cout << "MV Student-t with df="<<thetaMVt(0)<<" and inverse variance matrix of:"<<endl;
  ArrayXd Vinvlong=thetaMVt.tail(d*d);
  Map<MatrixXd> Vinv(Vinvlong.data(),d,d);
  cout << Vinv<<endl;
  l=l_MVt(x,thetaMVt);
  cout << l<< endl; // checked in R that this is the right answer

  ArrayXd thetaSkewNormal(d+1);
  thetaSkewNormal(0)=2; // alpha
  thetaSkewNormal.tail(d)=ArrayXd::LinSpaced(d,1,10); // Scale parameters
  cout << "Product of SkewNormals with [alpha,Sc1,...,Scd]=\n"<< thetaSkewNormal.matrix().transpose()<<endl;
  l=l_SkewNormalCpts(x,thetaSkewNormal);
  gl=gl_SkewNormalCpts(x,thetaSkewNormal);
  cout << l <<endl<<gl.transpose()<<endl;
  gl=glp_from_lp(x, thetaSkewNormal, &l_SkewNormalCpts);
  cout << gl.transpose()<<endl;

  // Check the ancillary functions at the boundaries
  ArrayXd ys=ArrayXd::LinSpaced(9,-9,-5);
  cout <<"PI="<<_PI<<endl;
  cout <<"xs="<<ys.matrix().transpose()<<endl;
  cout <<"logPhi xs"<<logPhi(ys).matrix().transpose()<<endl;
  cout <<"phi/Phi xs"<<phiOPhi(ys).matrix().transpose()<<endl;

  // Modified Rosenbrock
  ArrayXd thetaMR(d+1);
  thetaMR(0)=2; // beta
  thetaMR.tail(d/2)=ArrayXd::LinSpaced(d/2,1,10); // Scale parameters
  cout << "Product of Modified Rosenbrocks with [beta,Sc1,...,Scd]=\n"<< thetaMR.matrix().transpose()<<endl;
  l=l_ModifiedRosenbrock(x,thetaMR);
  gl=gl_ModifiedRosenbrock(x,thetaMR);
  cout << l <<endl<<gl.transpose()<<endl;
  gl=glp_from_lp(x, thetaMR, &l_ModifiedRosenbrock);
  cout << gl.transpose()<<endl;
  
  
  return 0;
}
*/

