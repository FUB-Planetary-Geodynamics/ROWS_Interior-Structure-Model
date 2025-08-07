# Author: Lena Noack
# GNU General Public License v3.0 (GPL-3.0)

import numpy as np
from values import *
from Help_functions import get_Tmelt_water
from water import getWater


def EOS_core(Pr, T, elem='FeSi',xelem=0.25):
  # Pr = 3 # pressure in GPa
  # T = 1600 # temperature in K
  # elem # name of species FeSi/FeC/FeS/FeO
  # xelem # array of molar fractions of name of species FeSi/FeC/FeS/FeO
  # 0.25 -> Fe3X1; 0.5 -> FeX1
  species_Fe = 'fehcp'
  species = ['Fe3C','FeSi','FeS','FeO']
  x = [0.25,0.5,0.5,0.5128]
  massO  = 15.999; massC  = 12.011; massS  = 32.065; massSi = 28.085; massFe = 55.845
  masses = [massC,massSi,massS,massO]
  mol = 6.02214076e23  # atoms

  if elem=='Fe':
    p = values(species_Fe)
    Cp,rho_Fe_T,alpha,KT,KS,mu,M,V,V0 = mineral(Pr,T,species_Fe)
    return Cp,rho_Fe_T,alpha,KT,KS,mu,V
  elif elem=='FeSi':
    jelem = 1
  elif elem=='FeC':
    jelem = 0
  elif elem=='FeS':
    jelem = 2
  elif elem=='FeO':
    jelem = 3
  else:
    print("ERROR - element not available")

  p = values(species[jelem])
  VFeX = mineral(Pr,300,species[jelem])[7]/(mol*1e-24*p.n)
  p = values(species_Fe)
  VFe = mineral(Pr,300,species_Fe)[7]/(mol*1e-24*p.n)
  rho_Fe = mineral(Pr,300,species_Fe)[1]
  Cp, rho_Fe_T, alpha, KT, KS, mu, M, V, V0 = mineral(Pr, T, species_Fe)
  VFeX_over_VFe = (VFeX/VFe-(1-x[jelem]))/x[jelem]
  drho_Fe_X = (1-xelem + xelem*masses[jelem]/massFe)/(1-xelem + xelem*VFeX_over_VFe)-1
  rho_Fe_X = drho_Fe_X*rho_Fe + rho_Fe
  rho_Fe_X = rho_Fe_X * rho_Fe_T/rho_Fe # scaled T effect of Fe EOS applied to FeX
  #return rho_Fe_X

  return Cp,rho_Fe_X,alpha,KT,KS,mu,[V]

def EOS_core_mixed(Pr, T, xC=0, xSi=0.25, xS=0, xO=0):
  # Pr = 3 # pressure in GPa
  # T = 1600 # temperature in K
  # x = 0.25 -> Fe3X1; 0.5 -> FeX1
  # multiple x>0: Fe3Si + Fe2O + ...
  # max values: xC=0.25, xSi=0.5, xS=0.5, xO=0.5128
  species_Fe = 'fehcp'
  species = ['Fe3C','FeSi','FeS','FeO']
  x = [0.25,0.5,0.5,0.5128]
  massO  = 15.999; massC  = 12.011; massS  = 32.065; massSi = 28.085; massFe = 55.845
  masses = [massC,massSi,massS,massO]
  mol = 6.02214076e23  # atoms

  if xC+xSi+xS+xO == 0: # pure iron
    p = values(species_Fe)
    Cp,rho_Fe_T,alpha,KT,KS,mu,M,V,V0 = mineral(Pr,T,species_Fe)
    return Cp,rho_Fe_T,alpha,KT,KS,mu,V
  else:
    # volumes of lighter species
    pC = values(species[0])
    pSi= values(species[1])
    pS = values(species[2])
    pO = values(species[3])
    VFeC = mineral(Pr,300,species[0])[7]/pC.n#/(mol*1e-24*pC.n)
    VFeSi= mineral(Pr,300,species[1])[7]/pSi.n#/(mol*1e-24*pSi.n)
    VFeS = mineral(Pr,300,species[2])[7]/pS.n#/(mol*1e-24*pS.n)
    VFeO = mineral(Pr,300,species[3])[7]/pO.n#/(mol*1e-24*pO.n)
    # iron values at 300K and at T
    p = values(species_Fe)
    VFe = mineral(Pr,300,species_Fe)[7]#/(mol*1e-24*p.n)
    #rho_Fe = mineral(Pr,300,species_Fe)[1]
    Cp, rho_Fe_T, alpha, KT, KS, mu, M, V, V0 = mineral(Pr, T, species_Fe)
    # volume ratios of theoretical light elements over Fe
    VFeC_VFe = (VFeC/VFe -(1-x[0]))/x[0]
    VFeSi_VFe= (VFeSi/VFe-(1-x[1]))/x[1]
    VFeS_VFe = (VFeS/VFe -(1-x[2]))/x[2]
    VFeO_VFe = (VFeO/VFe -(1-x[3]))/x[3]
    # combined density deficits, scaled T effect of Fe EOS applied to FeX
    rho_Fe_X = rho_Fe_T * ((1-xC-xSi-xS-xO + (xC*massC+xSi*massSi+xS*massS+xO*massO)/massFe)
                        /  (1-xC-xSi-xS-xO + (xC*VFeC_VFe+xSi*VFeSi_VFe+xS*VFeS_VFe+xO*VFeO_VFe)))

  return Cp,rho_Fe_X,alpha,KT,KS,mu,[V] # other parameters are taken from iron

def EOS_rock(Pr,T,arr_species,mf_vec,Store_V_vec=[]): # Pressure, Temperature, Array of Species, Mass Fractions Array
    # Example values for Earth:
    # Pr = 3 # pressure in GPa
    # T = 1600 # temperature in K
    # arr_species = ['fo', 'fa', 'jd']
    # mf_vec = [0.774, 0.126, 0.1] # mass fractions, sum needs to be 1

    Cp_vec = []
    alpha_vec = []
    rho_vec = []
    KT_vec = []
    KS_vec = []
    mu_vec = []
    M_vec = []
    V_vec = []
    V0_vec = []

    for i,species in enumerate(arr_species):
        if species=='h2o':
            #Cp = 4200
            #alpha = 0.00007
            #rho = 1000
            KT = 0.0
            KS = 0.0
            mu = 0.0
            M = 18.0

            [Tm,phase] = get_Tmelt_water(Pr*1e9)
            if T<Tm: # ice phase
                if phase == 10:
                    Cp, rho, alpha, KT, KS, mu, M, V, V0 = mineral(Pr, T, 'iceX')
                    mu = 0.0
                elif phase == 7:
                    Cp, rho, alpha, KT, KS, mu, M, V, V0 = mineral(Pr, T, 'iceVII')
                    mu = 0.0
                elif phase == 1: # simple estimate for now since ice shell is anyway very thin
                    Cp = 2090
                    alpha = 5.0e-5
                    rho = 920
                else:  # so far only phase 6, no surface ice included
                    Cp, rho, alpha, KT, KS, mu, M, V, V0 = mineral(Pr, T, 'iceVI')
                    mu = 0.0
            else:
                [rho, alpha, Cp, k, error] = getWater(Pr*1e9, T)

            V0 = M # M/rho[g/cm^3]; at 273.15K, 1 bar => rho = 1 g/cm^3
            V = M/rho

        elif Store_V_vec==[]:
            Cp,rho,alpha,KT,KS,mu,M,V,V0 = mineral(Pr, T, species) # Pressure, Temperature, Species
        else:
            if len(Store_V_vec)==len(arr_species):
                Cp,rho,alpha,KT,KS,mu,M,V,V0 = mineral(Pr, T, species,Store_V_vec[i]) # Pressure, Temperature, Species
                #print(species,Store_V_vec[i],V,V0)
            else: # i.e. different material layer than in previous iteration
                Cp,rho,alpha,KT,KS,mu,M,V,V0 = mineral(Pr, T, species) # Pressure, Temperature, Species
                
        Cp_vec.append(Cp)
        alpha_vec.append(alpha)
        rho_vec.append(rho)
        KT_vec.append(KT)
        KS_vec.append(KS)
        mu_vec.append(mu)
        M_vec.append(M)
        V_vec.append(V)
        V0_vec.append(V0)
    

    if len(arr_species)>1:
      # mineral assemblade
      # if len=1 -> use Cp etc. from above    
      Cp=0
      rho=0
      alpha=0
      KT_1=0
      KT_2=0
      KS_1=0
      KS_2=0
      mu_1=0
      mu_2=0

      for i in range(len(arr_species)):
          Cp += mf_vec[i]*Cp_vec[i]
          rho += mf_vec[i]/rho_vec[i]
      
      rho=rho**(-1)
      for i in range(len(arr_species)): 
          vf = mf_vec[i]*rho/rho_vec[i] # volume fraction
          alpha += vf*alpha_vec[i]
      
          if vf>0: 
              KT_1 = KT_1 + vf*KT_vec[i]
              KT_2 = KT_2 + vf/KT_vec[i]
              KS_1 = KS_1 + vf*KS_vec[i] 
              KS_2 = KS_2 + vf/KS_vec[i]
              if mu_vec[i]>0:
                  mu_1 += vf*mu_vec[i]
                  mu_2 += vf/mu_vec[i]
      
      KT = 0.5*KT_1 + 0.5/KT_2
      KS = 0.5*KS_1 + 0.5/KS_2
      if mu_2>0:
          mu = 0.5*mu_1 + 0.5/mu_2
      else:
          mu = 0
        
    return Cp,rho,alpha,KT,KS,mu,V_vec

#############################################################
#############################################################


def mineral(Pr, T, species, V_i=0): # Pressure, Temperature, Species

    p = values(species) # params for species
    V=get_V(Pr,T,p,V_i)
        
    if (p.EOS==2) or (p.EOS==3):
        val_BM3(V,T,p) # values for rho, Cp, alpha etc are stored in p
    elif p.EOS==1:
        val_Holz(V,T,p)
    else:
        val_Vinet(V,T,p)

    return p.Cp,p.rho,p.alpha,p.KT,p.KS,p.mu,p.M,p.V,p.V0

#############################################################
#############################################################

def get_V(Pr,T,p,V_i=0): # get V that fits to pressure Pr and temperature T

    tau = 0.1#0.01 # in bar  # CHECK: tau=0.1 works as well as 0.01 but is faster; tau=1 leads to deviation in results
    maxIter = 100

    if (V_i==0): # no V_i stored
        V_i = p.V0
        dV = 0.5*p.V0
    else:
        dV = 0
        #print(V_i)
        

    if p.EOS==3:
        P_i = P_BM3(p,V_i,T)
    elif p.EOS==2:
        p.KTp = 4.0 # third-order term will be zero
        P_i = P_BM3(p,V_i,T)
    elif p.EOS==1:
        P_i = P_Holz(p,V_i,T)
    else:
        P_i = P_Vinet(p,V_i,T)

    #print('------------------------ ',V_i,Pr,P_i)
        
    iter=1
    dir_old=0

    if dV == 0: # based on input V_i
        dV = p.V0 * 0.5 * abs(P_i-Pr)/max(P_i,Pr)  # if Pi, Pr very close together, then start with smaller dV
        
    while (np.abs(P_i-Pr) > tau) and (iter < maxIter): # bisection method
        iter=iter+1
        #print(iter,dV)
        if P_i>Pr:
            V_i = V_i+dV
            if dir_old!=1:
                dV = 0.5*dV
            dir_old=1 # continue to search in the same direction, dV stays the same
        else:
            V_i = V_i-dV
            if dir_old!=-1:
                dV = 0.5*dV
            dir_old=-1 # continue to search in the same direction, dV stays the same
        #dV = 0.7*dV
        
        if (p.EOS==3) or (p.EOS==2):
            P_i = P_BM3(p,V_i,T)
        elif p.EOS==1:
            P_i = P_Holz(p,V_i,T)
        else:
            P_i = P_Vinet(p,V_i,T)

    #print(P_i,Pr,p.V0,V_i,iter)
    return V_i

#############################################################
#############################################################
def P_BM3(p,V,T): # pressure for Birch-Murnaghan EOS of 3rd order
    
    R = 8.3144621

    x = p.V0/V
    f = 0.5*(x**(2/3) - 1.0)

    if (p.en==1): # for iron with light elements, use a different formulation based on Holzapfel iron bcp
      gamma=(p.gamma0-p.gammaInf)/x**p.beta + p.gammaInf # Bouchet et al. 2013 Eq. 7
      theta=p.theta0*x**(p.gammaInf) * np.exp((1-x**(-p.beta))*(p.gamma0-p.gammaInf)/p.beta) # Bouchet et al. 2013 Eq. 8
    else:
      if (1.0+6.0*f*p.gamma0 + 3.0*f**2*p.gamma0*(-2.0+6.0*p.gamma0-3.0*p.q0))<0:
        theta=p.theta0
      else:
        theta = p.theta0*np.sqrt(1.0+6.0*f*p.gamma0 + 3.0*f**2*p.gamma0*(-2.0+6.0*p.gamma0-3.0*p.q0)) # characterisitc vibrational temperature

      gamma = (1.0+2.0*f)*p.gamma0*(1.0+f*(-2.0+6.0*p.gamma0-3.0*p.q0))*(p.theta0/theta)**2
      if gamma==0:
        gamma=p.gamma0

    Eth = 3*p.n*R*T*debye(theta/T)
    Eth0 = 3*p.n*R*T*debye(theta/p.T0)
    E = gamma/V*(Eth-Eth0)
    Pr = 1.5*p.KT0*(x**(7.0/3.0)-x**(5.0/3.0))*(1+0.75*(p.KTp-4)*(x**(2.0/3.0)-1)) + E

    return Pr


#############################################################
def debye(thetaT):

    return 3/thetaT**3 * IntEth(thetaT)

#############################################################
def IntEth(x):
    # int_0^x  tau^3/(exp tau - 1) dtau
    
    nr = 100 # number of intergation steps
    dx = x/nr
    #IntVal = 0.0
    #tau = 0.5*dx
    #
    #for i in range(nr): # 1:nr-1
    #    IntVal = IntVal + dx*(tau**3/(np.exp(tau)-1))
    #    tau = tau + dx
    tau = dx * (np.arange(nr) + 0.5) # midpoint rule
    integrand = tau**3 / (np.exp(tau)-1)
    IntVal = np.sum(integrand)*dx

    return IntVal


#############################################################
def P_Holz(p,V,T): # pressure for Holzapfel EOS 
    
    R = 8.3144621

    # 1003.6 * (Z/V0[cm^3/mole])^(5/3) = 1003.6 * (Z/V0[mm^3/mole])^(5/3) * (1e-3)^(5/3) = ... * 1e-5
    PFG0 = 1003.6e5*(p.Z/(p.V0))**(5.0/3.0)
#    PFG0 = 0.10036e9*(p.Z/(p.V0))**(5.0/3.0) # 1003.6e9 -> 0.10036e9 due to V0*1000^5/3=value*10^5
    c0 = -np.log(3.0*p.KT0/PFG0)
    c2 = 3.0/2.0*(p.KTp - 3.0)-c0
    zeta=(V/p.V0)**(1.0/3.0) 
    zeta3=zeta**3

    theta=p.theta0*zeta3**(-p.gammaInf)* np.exp((1-zeta3**p.beta)*(p.gamma0-p.gammaInf)/p.beta)
    gamma=(p.gamma0-p.gammaInf)*zeta3**p.beta + p.gammaInf

    E = gamma/V*(3*R*(0.5*theta + theta/(np.exp(theta/T)-1)) - 3*R*(0.5*theta + theta/(np.exp(theta/p.T0)-1)))

    Pr = 3.0*p.KT0*zeta**(-5)*(1-zeta)*np.exp(c0*(1-zeta))*(1+c2*(zeta-zeta**2)) + E
    Pr = Pr + 1.5*R/V * p.m*p.a0*zeta**(3.0*p.m)*T**2
    
    return Pr
    
#############################################################
def P_Vinet(p,V,T): # pressure for Vinet EOS 
    
    f=(1/2)*(((p.V0/V)**(2/3))-1)
#    gamma0p=-1.94 # Grüneisenparameter derivative
    Avo=6.0221468*10**23 # Atome pro mol [mol^-1]
    a1=6*p.gamma0
    a2=(-12*p.gamma0)+(36*(p.gamma0**2))-(18*p.gamma0p)
    gamma=((2*f+1)*(a1+a2*f))/(6*(1+(a1*f)+((1/2)*a2*(f**2))))
#    E0=20.5953*(1.602176643*10**-22)*Avo # eV/atom in kJ/mol
    x =(V/p.V0)**(1.0/3.0)
    E =(9.0*p.KT0*p.V0/p.eta**2*(1.0+(p.eta*(1.0-x)-1.0)*np.exp(p.eta*(1.0-x)))) # Helmholtz energy
    E = gamma/V*(E-p.E0)
    Pr =(3.0*p.KT0*x**-2*(1.0-x)*np.exp(p.eta*(1.0-x))) + E

    return Pr

#############################################################
#############################################################
def val_BM3(V,T,p):
    
    R = 8.3144621
    x = p.V0/V
    f = 0.5*(x**(2/3) - 1.0) # finite volumetric strain
    
    if (p.en==1): # for iron with light elements, use a different formulation based on Holzapfel iron bcp
      gamma=(p.gamma0-p.gammaInf)/x**p.beta + p.gammaInf # Bouchet et al. 2013 Eq. 7
      theta=p.theta0*x**(p.gammaInf) * np.exp((1-x**(-p.beta))*(p.gamma0-p.gammaInf)/p.beta) # Bouchet et al. 2013 Eq. 8
      q=0
    else:
      if (1.0+6.0*f*p.gamma0 + 3.0*f**2*p.gamma0*(-2.0+6.0*p.gamma0-3.0*p.q0))<0:
        theta=p.theta0
      else:
        theta = p.theta0*np.sqrt(1.0+6.0*f*p.gamma0 + 3.0*f**2*p.gamma0*(-2.0+6.0*p.gamma0-3.0*p.q0)) # characterisitc vibrational temperature

      gamma = (1.0+2.0*f)*p.gamma0*(1.0+f*(-2.0+6.0*p.gamma0-3.0*p.q0))*(p.theta0/theta)**2
      if gamma==0:
        gamma=p.gamma0
      if gamma==0:
        q=0
      else:
        q=(-2*gamma+6*gamma**2 +(1+2*f)**2 *p.gamma0*(2-6*p.gamma0+3*p.q0)*(p.theta0/theta)**2)/(3*gamma)


    Cv = 3*p.n*R*(4*debye(theta/T) - (theta/T) * 3/(np.exp(theta/T)-1)) # molar isochore heat capacity
    Cv0 = 3*p.n*R*(4*debye(theta/p.T0) - (theta/p.T0) * 3/(np.exp(theta/p.T0)-1))

    # output values
    p.KT=(1+2*f)**(5/2)*p.KT0*(1+f*(-5+3*p.KTp)+27/2*f**2 *(-4+p.KTp)) # bulk modulus
    p.KT=p.KT+(-gamma**2/V*(Cv*T-Cv0*p.T0)+gamma/V*(1-q+gamma)*(3*p.n*R*T*debye(theta/T)-3*p.n*R*p.T0*debye(theta/p.T0)))
    p.alpha=gamma*Cv/(p.KT*V) # thermal expansion coefficient
    p.KS=p.KT*(1+p.alpha*gamma*T)
    p.Cp=Cv*(1+p.alpha*gamma*T) / (1e-3*p.M) # divide by mol mass; specific isobaric heat capacity
    p.rho=1e6*p.M/V # density M [g mol^-1], V [mm^3 mol^-1] * 10^6 --> rho [kg m^-3]
    p.V=V
    
    etaS=-gamma-1/2*(theta/p.theta0)**2 *(2*f+1)**2 * (-2*p.gamma0-2*p.eta)
    p.mu = (1+2*f)**(5/2)*(p.G0+(3*p.KT0*p.Gp-5*p.G0)*f+(6*p.KT0*p.Gp-24*p.KT0-14*p.G0+9/2*p.KT0*p.KTp)*f**2)
    p.mu = p.mu - etaS/V*(3*p.n*R*T*debye(theta/T)-3*p.n*R*p.T0*debye(theta/p.T0))


    return    
    
#############################################################
def val_Holz(V,T,p):
    
    R = 8.3144621
    M=p.M*1e6
    
    PFG0 = 0.10036e9*(p.Z/p.V0)**(5.0/3.0)
    c0 = -np.log(3.0*p.KT0/PFG0)
    c2 = 3.0/2.0*(p.KTp - 3.0)-c0
    zeta=(V/p.V0)**(1.0/3.0)  #zetax(p,T,errorCode)
    zeta3=zeta**3 # V/V0
    gamma=(p.gamma0-p.gammaInf)*zeta3**p.beta + p.gammaInf # Bouchet et al. 2013 Eq. 7
    theta=p.theta0*zeta3**(-p.gammaInf) * np.exp((1-zeta3**p.beta)*(p.gamma0-p.gammaInf)/p.beta) # Bouchet et al. 2013 Eq. 8
    Cv=3.0*R*theta**2 * np.exp(theta/T)/( (-1.0+np.exp(theta/T))**2 * T**2 )
    Cv0=3.0*R*theta**2 * np.exp(theta/p.T0)/( (-1.0+np.exp(theta/p.T0))**2 * p.T0**2 )
    Eth=3.0*R*(theta/2.0 + theta/(np.exp(theta/T)-1.0))
    Eth0=3.0*R*(theta/2.0 + theta/(np.exp(theta/p.T0)-1.0))
    dEeadx = 1.5*R*p.m**2*p.a0*zeta3**(p.m-1) * T**2 # anharmonic and electronic terms pressure Bouchet et al. 2013 Eq. 9, 10
    dEeadx0 = 1.5*R*p.m**2*p.a0*zeta3**(p.m-1) * p.T0**2
    Eea = 1.5*R*p.m*p.a0*zeta3**p.m * T**2
    Eea0 = 1.5*R*p.m*p.a0*zeta3**p.m * p.T0**2
    KT=(np.exp(c0-c0*zeta)*p.KT0*(5.0+zeta*(-4.0+2.0*c2*(-2.0+zeta)*(-1.0+zeta)+c0*(-1.0+zeta)*(-1.0+c2*(-1.0+zeta)*zeta))))/zeta**5
    KT=KT+(-(gamma**2 *(T*Cv-p.T0*Cv0))+(gamma*(1.0-p.beta+gamma)+p.beta*p.gammaInf)*(Eth-Eth0))/V
    p.KT=KT-(dEeadx-dEeadx0)/p.V0+(Eea-Eea0)/V
    p.Cv=Cv+3.0*R*p.m*p.a0*zeta3**p.m * T
    p.alpha=gamma*p.Cv/(KT*V) # Grüneisen parameter gamma = (alpha*KT*V) / Cv Bouchet et al. 2013 Eq. 2
    p.KS=p.KT*(1+p.alpha*gamma*T)
    p.Cp=p.Cv*(1.0+p.alpha*gamma*T) / (M*1.0e-9)  # from J/mol K to J/kg K: division by mol mass
    p.rho=M/V
    p.V=V    
    p.mu=0 # no shearing if liquid core

    return    

#############################################################
def val_Vinet(V,T,p):

    R = 8.3144621
    
    return    
    
    
    
    