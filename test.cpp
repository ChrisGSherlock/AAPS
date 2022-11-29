#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <chrono>
#include <random>
//#include "../Targets/toy_targets.h"
#include "toy_targets.h"
#include "AAPS.h"

// Test bed for AAPS - four types of target, any dimension up to 999,
// any eccentricity up to 999, different types of spacing.
// Any combination of tuning parameters.

using namespace std;
using namespace Eigen;


static int CheckInputs(int targ, int nits, int prt, int thin, int precon, int alg, double par1, int par2) {
  int bad=0;

  if (nits<0) {
    cout << "Bad nits: "<<nits<<endl; bad=1;
  }
  if (prt<0) {
    cout << "Bad prt: "<<prt<<endl; bad=1;
  }
  if (thin<1) {
    cout << "Bad thin: "<<thin<<endl; bad=1;
  }
  if ((precon<0) || (precon>1)) {
    cout << "Bad precon: "<<precon<<endl; bad=1;
  }
  if ((alg<1) || (alg>3)) {
    cout << "Bad weighting scheme: "<<alg<<endl; bad=1;
  }
  if (par1<0) {
    cout << "Bad epsilon: "<<par1<<endl; bad=1;
  }
  if (par2<0) {
    cout << "Bad K: "<<par2<<endl; bad=1;
  }

  if (!bad) {
    cout << "targ="<<targ<<", nits="<<nits<<", prt="<<prt<<", thin="<<thin<<", precon="<<precon<<", alg="<<alg<<", par1="<<par1<<", par2="<<par2<<endl;
  }
  
  return bad;
}
static int CheckType(const ArrayXi &type) {
  int bad=0;

  if (type(1)<1) {
    cout << "Bad d: "<<type(1)<<endl; bad=1;
  }
  if (type(3)<1) {
    cout << "Bad ecc: "<<type(3)<<endl; bad=1;
  }

  return bad;
}


// test out the algorithm
int main(int argc, const char** pargv)  {
  int targ=0, nits=1, thin=1, prt=1, precon=0, Wtype=3, i=1, bad;
  double par1=0.0; // epsilon
  int par2=0; // K

  if (argc==1) {
    cout << pargv[0] <<" targ(0) nits(1) prt(1) thin(1) precon(0) Wtype(3) eps(0.0) K(0)\n";
    cout << "Defaults in brackets.\n";
    cout << "Targs:\n- 0=10d Gaussian with SDs 1-5;\n- 1=50d Gaussian with diag Hess 1-10.\n- General Ddddseee where D=distribution (0=Gaussian, 1=logistic, 2=SkewedGaussian, 3=ModifiedRosenbrock), d=dimension, s=scale spacing (0=SD, 1=Var, 2=invSD, 3=Hess, +4 if jittered, 8=RadfordNeal-like)\n";
    cout << "nits=#iterations, print every *prt* iterations, thin=thinning factor.\n";
    cout << "precon=0 => no preconditioning, precon=1 => full preconditioning.\n";
    cout << "Wtype: 1=pitil, 2=||x'-x||^2, 3=pitil||x'-x||^2.\n";
    cout << "K=0 is single segment; K>0 is ruler of length K+1.\n";
    return 0;
  }
  
  if (argc>1) { 
    targ=atoi(pargv[i++]);
  }
  if (argc>2) {
    nits=atoi(pargv[i++]);
  }
  if (argc>3) {
    prt=atoi(pargv[i++]);
  }
  if (argc>4) {
    thin=atoi(pargv[i++]);
  }
  if (argc>5) {
    precon=atoi(pargv[i++]);
  }
  if (argc>6) {
    Wtype=atoi(pargv[i++]);
  }
  if (argc>7) {
    par1=atof(pargv[i++]); // epsilon
  }
  if (argc>8) {
    par2=atoi(pargv[i++]); // K.
  }

  bad=CheckInputs(targ,nits,prt,thin,precon,Wtype,par1,par2);
  if (bad==1) return 0;
  
  std::random_device rd;  //Will use to obtain seed for random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> StdUnif(0.0, 1.0);
  std::normal_distribution<double> StdNormal(0.0, 1.0);

  // Create suitable variables for all algorithms
  ArrayXd x0, thetall, thetapri, tmp, Scales, Sigma;
  ArrayXi targ_type(4);
  ArrayXd Mdiag;  // diagonal preconditioning for HMC 
  string targstr;
  Targd prior, likelihood;
  double max_ecc;
  int d;

  if (targ==0) { // 10d Gauss with SDs linear from 1 to 5
    targ_type<<0,10,0,5;
    cout << "10d Gaussian with SD linear from 1 to 5\n";
  }
  else if (targ==1) { // 50d Gaussian with diag Hess from 1 to 100.
    targ_type<<0,50,2,10;
    cout << "50d Gaussian with diag Hess linear from 1 to 100\n";
  }
  else {
    // Ddddseee   D=distribution, d=dimension, s=scale spacing, e=eccentricity
    
    targ_type(0) =  targ / 10000000;
    targ_type(1) = (targ - targ_type(0)*10000000)/10000;
    targ_type(2) = (targ - targ_type(0)*10000000 - targ_type(1)*10000)/1000;
    targ_type(3) =  targ - targ_type(0)*10000000 - targ_type(1)*10000 - targ_type(2)*1000;
  }

  bad=CheckType(targ_type);
  if (bad==1) return 0;
  
  d = targ_type(1);
  max_ecc = targ_type(3) * 1.0;
  targstr = "D" + to_string(d) + "_E"+to_string(targ_type(3));

  if ((targ_type(2)>=0) && (targ_type(2)<=3)) {
    if(targ_type(2) == 0){
      targstr.append("_SDlinspc");  
      tmp = Eigen::ArrayXd::LinSpaced(d,1.0,max_ecc);
      Sigma = tmp*tmp;
      
    } else if(targ_type(2) == 1){
      targstr.append("_VARlinspc");
      tmp = Eigen::ArrayXd::LinSpaced(d,1.0,max_ecc*max_ecc);
      Sigma = tmp;
      
    } else if(targ_type(2) == 2){
      targstr.append("_Hlinspc");
      tmp = Eigen::ArrayXd::LinSpaced(d,1.0,max_ecc*max_ecc);
      Sigma = (max_ecc*max_ecc)/tmp;
      
    } else if(targ_type(2) == 3){
      targstr.append("_invSDlinspc");
      tmp = Eigen::ArrayXd::LinSpaced(d,1.0,max_ecc);
      Sigma = (max_ecc*max_ecc)/tmp/tmp;
    }
    
    std::sort(Sigma.data(),Sigma.data()+Sigma.size());
    Scales=Sigma.sqrt();
      
  } else if(targ_type(2)>=4 && targ_type(2)<=7){
    std::mt19937 fixedgen(1234567);
    double ddm1=(double)d-1;
    tmp = Eigen::ArrayXd::LinSpaced(d,0.0,ddm1); // 0 to d-1
    for(int cpt=1;cpt<d-1; cpt++){
      tmp(cpt) += StdUnif(fixedgen)-0.5;
    }
    tmp=tmp/ddm1; // now from 0 to 1 with the right jitter
      
    if(targ_type(2) == 4){
      targstr.append("_SDunif");   
      Scales=tmp*(max_ecc-1)+1;
      Sigma=Scales*Scales;
	
    } else if(targ_type(2) == 5){
      targstr.append("_VARunif");
      Sigma=tmp*(max_ecc*max_ecc-1)+1;
      Scales=Sigma.sqrt();
	
    } else if(targ_type(2) == 6){
      targstr.append("_Hunif");
      double oneoxi2=1/max_ecc/max_ecc;
      tmp=oneoxi2+tmp*(1-oneoxi2); // Hess
      Sigma=1/tmp;
      Scales=Sigma.sqrt();

    } else if(targ_type(2) == 7){
      targstr.append("_invSDunif");
      double oneoxi=1/max_ecc;
      tmp=oneoxi+tmp*(1-oneoxi);
      Scales=1/tmp;
      Sigma=Scales*Scales;
    }
    
    std::sort(Scales.data(),Scales.data()+Scales.size());
    std::sort(Sigma.data(),Sigma.data()+Sigma.size());
  }
  //  cout <<"Scales\n";
  //  cout <<Scales.matrix().transpose()<<endl;
  if(targ_type(0) == 0){
            
    targstr.insert(0,"Gauss_");

    ArrayXd z=StdNormalAXd(d,gen,StdNormal);
    x0 = Scales*z;
    thetall=ArrayXd::Zero(2*d);
    thetall.tail(d) = Sigma;
    prior.theta=ArrayXd::Zero(1); // any value as is irrelevant
    prior.pl_fn=&l_null;
    prior.pgl_fn=&gl_null;
    likelihood.theta=thetall;
    likelihood.pl_fn=&l_GaussDiag;
    likelihood.pgl_fn=&gl_GaussDiag;
  } else if(targ_type(0) == 1){
    targstr.insert(0,"Logis_");
    ArrayXd z=StdLogisticAXd(d,gen,StdUnif);
    x0 = Scales*z;
    thetall=ArrayXd::Zero(2*d);
    thetall.tail(d) = Scales;
    
    prior.theta=ArrayXd::Zero(1); // any value as is irrelevant
    prior.pl_fn=&l_null;
    prior.pgl_fn=&gl_null;
    likelihood.theta=thetall;
    likelihood.pl_fn=&l_LogisticCpts;
    likelihood.pgl_fn=&gl_LogisticCpts;

  } else if(targ_type(0) == 2){ 
    double omega=3.0; // Skewness parameter hardwired to 3
    targstr.insert(0,"SkewN_");  
    ArrayXd z=StdSkewNormalAXd(d,omega,gen,StdNormal); 
    x0 = Scales*z;
    thetall=ArrayXd::Zero(d+1);
    thetall.tail(d) = Scales;
    thetall(0)=omega;

    prior.theta=ArrayXd::Zero(1); // any value as is irrelevant
    prior.pl_fn=&l_null;
    prior.pgl_fn=&gl_null;
    likelihood.theta=thetall;
    likelihood.pl_fn=&l_SkewNormalCpts;
    likelihood.pgl_fn=&gl_SkewNormalCpts;

  } else if (targ_type(0) == 3){
    if ((d/2)*2 != d) {
      cout << "Modified Rosenbrock needs an even d, not "<<d<<endl;
      exit(1);
    }
    double beta=1.0;
    Scales=Eigen::ArrayXd::LinSpaced(d/2,0.0,(double)(d/2)-1);
    if (d==2) {
      Scales=Scales+1;
    } else {
      Scales=1+(max_ecc*max_ecc-1)*Scales/(d/2-1);
      Scales=sqrt(Scales);
    }
    thetall=ArrayXd::Zero(d/2+1); 
    thetall.tail(d/2) = Scales;
    thetall(0)=beta;
    targstr.insert(0,"ModRos_");  
    x0=ModifiedRosenbrockAXd(d,gen,StdNormal,thetall);
    
    prior.theta=ArrayXd::Zero(1); // any value as is irrelevant
    prior.pl_fn=&l_null;
    prior.pgl_fn=&gl_null;
    likelihood.theta=thetall;
    likelihood.pl_fn=&l_ModifiedRosenbrock;
    likelihood.pgl_fn=&gl_ModifiedRosenbrock;
  }	

  cout << targstr << "\n";

  if (precon && (targ_type(0)<3)) { // not coded precon for MR
    Mdiag=1/Sigma;
  }
  else {
    Mdiag=ArrayXd::Constant(d,1.0);
  }

  cout <<"Wtype="<<Wtype<<", K="<<par2<<endl;

  aaps(nits, x0, Wtype, par1, par2, Mdiag, prior, likelihood,
       targstr,"Output/", thin, prt);

  return 0;
}
