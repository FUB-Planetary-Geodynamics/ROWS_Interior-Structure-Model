# Author: Lena Noack
# GNU General Public License v3.0 (GPL-3.0)

import numpy as np


#Get melting temperature:
#Temperature at pressure P [Pa] of 40% solidifed melt, used to define initial temperature profile of mantle and core

def get_Tmelt_MO(P_Pa):

    P_GPa = P_Pa*1e-9
    ratio = 0.4 # 0-solidus, 1-liquidus

    #melt law for P_GPa <= 10 # Morschhauser et al. 2011 (Takahashi 1990)
    T_liq = 2035.15+(57.46*P_GPa)-(3.487*(P_GPa**2))+(0.0769*(P_GPa**3))
    T_sol = 1409.15+(134.2*P_GPa)-(6.581*(P_GPa**2))+(0.1054*(P_GPa**3))
    T_melt_MO1 = min([5000,T_sol + ratio*(T_liq-T_sol)]) # to avoid errors for large pressures
    #melt law for 10 < P_GPa <= 100
    T_liq = 1980+(36.918*P_GPa)-(0.065444*(P_GPa**2))+(7.6686*1e-5*(P_GPa**3))-((3.09272*1e-8)*(P_GPa**4))
    T_sol = 1835+(36.918*P_GPa)-(0.065444*(P_GPa**2))+(7.6686*1e-5*(P_GPa**3))-((3.09272*1e-8)*(P_GPa**4))
    T_melt_MO2 = T_sol + ratio*(T_liq-T_sol)
    #melt law for P_GPa >100 [GPa]
    T_liq=5400*(P_GPa/140)**0.480
    T_sol=T_liq*(1-np.log(0.79))**(-1)
    T_melt_MO3 = max([1600,T_sol + ratio*(T_liq-T_sol)]) # to avoid errors for small pressures

    p_trans1 = 10
    p_trans2 = 100

    fac1=10
    fac2=0.05
    T_melt_MO = T_melt_MO1 + (T_melt_MO2-T_melt_MO1)*(np.arctan((P_GPa-p_trans1)*fac1)+np.pi/2)/np.pi + (T_melt_MO3-T_melt_MO2)*(np.arctan((P_GPa-p_trans2)*fac2)+np.pi/2)/np.pi

    return T_melt_MO

def get_Tmelt_core(P_Pa):
    # inner core melt temperature without any light elements, Stixrude et al., 2014
    P_GPa = P_Pa*1e-9
    T_melt_core = 6500 * (P_GPa/340)**0.515
    return T_melt_core

def get_Tmelt_water(P_Pa):
    P_MPa = P_Pa*1e-6
    Tmelt = 0
    phase = 0

    #get melt temperature depending on pressure
    if (P_MPa < 0.000611657):  # Ice I
      Tmelt = 10000.0 # no melting temperature for these low pressures, chose high number so that T<T_melt in any case
      phase = 1
    elif (P_MPa < 209.9): # water/Ice I
      tau = P_MPa/0.000611657 - 1.0
      Tmelt = 273.16*(1.0-0.00000016404*tau-0.000000000000207956*tau**2)
      phase = 1
    elif (P_MPa < 350.1): # water/Ice III
      tau = P_MPa/209.9 - 1.0
      Tmelt = 251.165*(1.0+0.052355*tau-0.0528534*tau**2+0.02846717*tau**3)
      phase = 3
    elif (P_MPa < 632.4): # water/Ice V
      tau = P_MPa/350.1 - 1.0
      Tmelt = 256.164*(1.0+0.1046*tau-0.0342*tau**2+0.009193711*tau**3)
      phase = 5
    elif (P_MPa < 2216.0): # water/Ice VI
      tau = P_MPa/632.4 - 1.0
      Tmelt = 273.31*(1.0+0.19065*tau-0.043492*tau**2+0.005985177*tau**3)
      phase = 6
    elif (P_MPa < 12834.0): # water/Ice VIIa
      tau = P_MPa/2216.0 - 1.0
      Tmelt = 355.0*(1.0+0.55259685*tau-0.17857832*tau**2+0.038338135*tau**3-0.0031581355*tau**4)
      phase = 7
    elif (P_MPa < 43647.0): # water/Ice VIIb
      tau = P_MPa/12834.0 - 1.0
      Tmelt = 750.0*(1.0+1.2122*tau-0.46171*tau**2+0.064497*tau**3)
      phase = 7
    else: # water/Ice X
      tau = P_MPa/43647.0 - 1.0
      Tmelt = 1630.0*(1.0+3.1544*tau-11.36*tau**2+20.798*tau**3-17.9125*tau**4+5.800116*tau**5)
      phase = 10

    return Tmelt,phase

def getphase(T,p,debug=0): # http://www.iapws.org/relguide/MeltSub.html
    # input:
    #   T in K
    #   p in Pa
    # return:
    #   getphase
    #   0 error, -2 gas, -1 liquid, -3 beyond critical point
    #   1,3,5,6,7,10 solid ice phase,
    #   999 ice no phase determined

    # Ice phase assuming borders between ih,III,V,VI are horizontal extrapolation of Triple Points as given in IAPWS Meltingcurves p 5, http://www1.lsbu.ac.uk/water/phase.html for ice 10

    getphase = 0
    av = [0.119539337e+7,0.808183159e+5,0.333826860e+4]
    bv = [0.300000e+1,0.257500e+2,0.103750e+3]
    psub = 0.0
    pih = 0.0
    p3 = 0.0
    p5 = 0.0
    pm = 0.0
    tm = 0.0

    if (273.31<= T) & (T <355.): # Ice VI
      Tref = 273.31 #K
      pref = 632.4e+6 # Pa
      Trel = T/Tref
      prel = 1 - 1.07476*(1-Trel**(4.6))
      pm = prel*pref
      if (pm < p):
        getphase = 6
        if ( p > (2.1e+9 + 1.16e+8*(T-278.0)/77.0) ):
            getphase = 7 # following diagram http://www1.lsbu.ac.uk/water/phase.html
        #if ( p > 636.4e+6):
        #    getphase = 7
        if (p > 47.E+9):
            getphase = 10
      else:
        getphase = -1
    elif (T >= 355.) & (T < 750): # derived from Schlager et al., 2004
      Tref = 355
      pref = 2.216E+9
      Trel = T / Tref
      prel = p / pref
      pm = pref * (2.478679-4.181254*Trel + 2.721316*Trel**2 )
      if (p < pm):
        getphase = -1
      else:
        getphase = 7
        if (p > 47.E+9):
            getphase = 10

    elif (T >= 750.) & (T < 1630.): # derived from Schlager et al., 2004
      Tref = 750
      pref = 12.834E+9
      Trel = T / Tref
      prel = p / pref
      pm = pref * (-2.98002 + 8.615783*Trel - 6.340367*Trel**2 + 1.714861*Trel**3)
      if (p < pm):
        getphase = -1
      else:
        getphase = 7
        if (p > 47.E+9):
            getphase = 10

    elif (T >= 1630.) & (T <= 2410.): #derived from Schlager et al., 2004
      Tref = 1630
      pref = 43.647E+9
      Trel = T / Tref
      prel = p / pref
      pm = pref * np.exp(0.01172*Trel**(10.39769))
      if (p < pm):
        getphase = -1
      else:
        getphase = 10
    else: # sublimation Pressure
      if (T <=273.16):
        Tref = 273.16 #K
        pref = 611.657 #Pa
        asv = [-0.212144006e+2,0.273203819e+2,-0.610598130e+1]
        bsv = [0.333333333e-2,0.120666667e+1,0.170333333e+1]
        Trel = T/Tref
        prel = 1/Trel*(asv[0]*Trel**(bsv[0])+asv[1]*Trel**(bsv[1])+asv[2]*Trel**(bsv[2]))
        prel = np.exp(prel)
        psub = prel*pref

      if(251.165 <= T) & (T <= 273.16): # Ice Ih
        Tref = 273.16 #K
        pref = 611.657 # Pa
        Trel = T/Tref
        prel = 1
        for i in range(3):
          prel = prel + av[i]*(1-Trel**(bv[i]))
        pih = prel*pref

      if (251.165<= T) & (T <= 256.164): #Ice III
        Tref = 251.165 #K
        pref = 208.566e+6 #Pa
        Trel = T/Tref
        prel = 1 - 0.299948*(1 - Trel**60)
        p3 = prel*pref

      if (256.164<= T) & (T <=273.31): #Ice V
        Tref = 256.164 #K
        pref = 350.1e+6 #Pa
        Trel = T/Tref
        prel = 1 - 1.18721*(1 - Trel**8)
        p5 = prel*pref

      if (273.16 <= T) & (T < 273.31) & (p > 611.657): # between 273.31  273.16 fulid or ice 5
        if (p5 == 0.0):
          getphase = 0        #error
        elif (p < p5): # check phase
          getphase = -1       #fluid
          print(p5)
        else:
          getphase = 999      #solid

      elif (273.31 >= T) & (T > 256.164) & (p > 611.657): # pressure higher than triple point
        if (p5 == 0.0) | (pih == 0.0):
          getphase = 0
        elif (p < p5) & (p > pih):
          getphase = -1
        elif (p < psub ):
          getphase = -1
        else:
          getphase = 999

      elif (256.164 >= T) & (T > 251.165) & (p > 611.657):
        if (p3 ==0.0) | (pih == 0.0) | (psub == 0.0):
          getphase = 0
        elif (p < p3) & (p > pih):
          getphase = -1
        elif (p < psub ):
          getphase = -1
        else:
          getphase = 999

      elif (T <= 273.16) & (p < 611.657):
        if (psub == 0.0):
          getphase = 0
        elif (p < psub):
          getphase = -1
        else:
          getphase = 999

      elif (T <= 251.16) & (p > 611.657):
        getphase = 999

      else:
          getphase = 0

    if (getphase == 999): # if phase is ice, then calculate the ice phase by aprox with triple point pressure
      if (p < 208566000):
        getphase = 1
      elif (p < 350e+6):
        getphase = 3
      elif (p < 632.4e+6):
        getphase = 5
      elif ( p < 2216e+6):
        getphase = 6
      elif ( p < 47e+9):
        getphase = 7
      elif(p >= 47e+9):
        getphase = 10
      else:
        getphase = 0

    if ( getphase == -1): # calculate dew-point http://www1.lsbu.ac.uk/water/phase.html
      dew = -2836.5744/(T**2) - 6028.076559/T + 19.54263612 - 0.02737830188*T + 1.6261698E-5*T**2
      dew = dew + 7.0229056E-10*T**3 - 1.8680009E-13*T**4 + 2.7150305*np.log(T)
      dew = np.exp(dew)

      if (p > 22064000) & (T > 647.096): # if above critical points
        getphase = -3
      elif (T > 251.156) & (T < 647.096) & (p > dew): # liquid
        getphase = -1
      else:       # vapor
        getphase = -2

    if (debug >= 1):
        print("Phase = ", getphase)

    return(getphase)

def GetHelmholtz(rho,T,debug):     # subroutine to calculate Helmholtz energy and derivatives
    # input:
    #   rho,T
    #   debug    # 0 no output, 1 iteration steps, 2 thermodynamic properties, 3 residuals
    # return:
    #   HE       # dictionary

    # Reference values and constants
    rhoref = 322.0 #kg/m^3
    Tref = 647.096 #K
    pref = 22.064e+6 #Pa
    kref = 1.0e-3 #W/mK
    muref = 1.0e-6 #Pa s
    R = 461.51805 #J/kg K
    Trel = T/Tref
    tau = Tref/T
    rhorel=rho/rhoref

    if (debug >= 4):
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print("++         Helmholtz free energy Part1: ideal gas part         ++")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print("rhorel=", rhorel)
      print("tau=", tau)

    # dimension(3,8) #!line-> i;n0i;gamma0i	- Data from Table1,i; -nodata=0
    A2 = np.array([[1.,-8.3204464837497,0.],
          [2.,6.6832105275932,0.],
          [3.,3.00632,0.],
          [4.,0.012436,1.28728967],
          [5.,0.97315,3.53734222],
          [6.,1.27950,7.74073708],
          [7.,0.96956,9.24437796],
          [8.,0.24873,27.5075105]]).transpose()

    if (debug >= 4):
        print("++           Helmholtz free energy ideal gas part              ++")

    phi_ideal = np.log(rhorel) + A2[1,0] +A2[1,1] * tau + A2[1,2]*np.log(tau) # Eq[5]
    for i in range (3,8):
      phi_ideal = phi_ideal + A2[1,i]*np.log(1-np.exp((-1)*tau*A2[2,i]))

    if (debug >= 4):
        print("phi_ideal = ", phi_ideal)

        print("++       Helmholtz free energy ideal gas part derivatives      ++")
        #http://www.iapws.org/relguide/IAPWS95-Rev.pdf Table 4.

    phi_ideal_r = 1 / rhorel
    phi_ideal_rr = -1 / rhorel**2
    phi_ideal_t = A2[1,1] + A2[1,2] / tau
    for i in range(3,8):
        phi_ideal_t = phi_ideal_t + A2[1,i]*A2[2,i]*((1/(1-np.exp((-1)*tau*A2[2,i]))-1))
    phi_ideal_tt =  -A2[1,2] / tau**2

    for i in range(3,8):
        phi_ideal_tt = phi_ideal_tt - A2[1,i]*(A2[2,i]**2)*np.exp((-1)*tau*A2[2,i])*1/((1-np.exp((-1)*tau*A2[2,i]))**2)
    phi_ideal_rt = 0

    if (debug >= 4):
      print("phi_ideal_r = ", phi_ideal_r)
      print("phi_ideal_rr = ",  phi_ideal_rr)
      print("phi_ideal_t = ",  phi_ideal_t)
      print("phi_ideal_tautau = ",  phi_ideal_tt)
      print("phi_ideal_rtau = ",  phi_ideal_rt)

      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print("++         Helmholtz free energy Part2: residual part          ++")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # Matrix data from http://twt.mpei.ac.ru/mcs/worksheets/iapws/coefs/IAPWS95-r.csv
    # numeric values for residual part
    # matrix was separated into integer and real vectors due to performance reasons
    A3C = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,4,6,6,6,6] # second column c(i)
    A3D = [1,1,1,2,2,3,4,1,1,1,2,2,3,4,4,5,7,9,10,11,13,15,1,2,2,2,3,4,4,4,5,6,6,7,9,9,9,9,9,10,10,12,3,4,4,5,14,3,6,6,6] # third column d(i)
    A3T = [-0.5,0.875,1.,0.5,0.75,0.375,1.,4.,6.,12.,1.,5.,4.,2.,13.,9.,3.,4.,11.,4.,13.,1.,7.,1.,9.,10.,10.,3.,7.,10.,
           10.,6.,10.,10.,1.,2.,3.,4.,8.,6.,9.,8.,16.,22.,23.,23.,10.,50.,44.,46.,50.] #fourth column t(i)

    A3N = [0.0125335479,7.8957634723,-8.7803203304,0.3180250935,-0.2614553386,-0.0078199752,0.0088089493,-0.6685657231,0.2043381095,
           -6.6212605039687E-005,-0.1923272116,-0.25709043,0.1607486849,-0.0400928289,3.9343422603254E-007,-7.5941377088144E-006,
           0.0005625098,-1.5608652257135E-005,1.1537996422951E-009,3.6582165144204E-007,-1.3251180074668E-012,-6.2639586912454E-010,
           -0.1079360091,0.017611491,0.2213229517,-0.4024766976,0.5808339999,0.0049969147,-0.0313587007,-0.7431592971,0.4780732992,
           0.0205279409,-0.1363643511,0.0141806344,0.0083326505,-0.029052336,0.0386150856,-0.0203934865,-0.001655405,0.0019955572,
           0.0001587031,-1.638856834253E-005,0.0436136157,0.0349940055,-0.0767881978,0.0224462773,-6.2689710414685E-005,-5.5711118565645E-010,
           -0.1990571835,0.3177749733,-0.1184118243] # fifth column n(i)

    # numeric values for i = 52 to 54 merged into integer 5x3 matrix and real 2x3
    #    real(dp),::A4_int
    #   real(dp),::A4_real
    #   real(dp),::A5_int
    #   real(dp),::A5_real

    #dimension(5,3)
    A4_int = np.array([[3,0,20,150,1],[3,1,20,150,1],[3,4,20,250,1]]).transpose() #    (d(i),t(i),apha(i),beta(i),ephion(i)
    #dimension(2,3)
    A4_real = np.array([[-0.31306260323435E2,1.21],[0.31546140237781E2,1.21],[-0.25213154341695E4,1.25]]).transpose() # n(i),gamma(i)
    #dimension(2,2)
    A5_int = np.array([[28,700],[32,800]]).transpose() # columns C,D
    #dimension(6,2)
    A5_real = np.array([[3.5,0.85,0.2,-0.14874640856724,0.32,0.3],[3.5,0.95,0.2,0.31806110878444,0.32,0.3]]).transpose() # columns a,b,B,n,A,beta

    #####################  phi_res ######################

    phi_res = 0

    for i in range(7):              # first sum in formula 6
        phi_res = phi_res + A3N[i] * (rhorel)**(A3D[i]) * tau**(A3T[i])

    for i in range(7,51):           # second sum
        phi_res = phi_res + A3N[i] * (rhorel)**(A3D[i]) *tau**(A3T[i]) * np.exp(-(rhorel**A3C[i]))

    for i in range(3):              # third sum
        phi_res = (phi_res + A4_real[0,i] * rhorel**(A4_int[0,i]) * tau**(A4_int[1,i])
                   * np.exp(-(A4_int[2,i]) * (rhorel-A4_int[4,i])**2 - A4_int[3,i] * (tau - A4_real[1,i])**2))

    for i in range(2):              # fourth sum debug here
        Theta = (1-tau) + A5_real[4,i]*((rhorel-1)**2)**(1/(2*A5_real[5,i]))
        Delta = (Theta)**2 + A5_real[2,i]*((rhorel-1)**2)**(A5_real[0,i])
        psi = np.exp(-A5_int[0,i]*(rhorel-1)**2-A5_int[1,i]*(tau-1)**2)
        phi_res = phi_res + A5_real[3,i]*(Delta)**(A5_real[1,i])*rhorel*psi

    if (debug >= 4):
        print("phi_res = ", phi_res ) # error to online calc <10E-4 http://twt.mpei.ac.ru/mcs/worksheets/iapws/IAPWS95.xmcd
        print("++++++   Helmholtz free energy residual part derivatives   ++++++")


    ########################  phi_res_r ######################

    phi_res_r = 0

    for i in range(7):              # first sum in formula 6
        phi_res_r = phi_res_r + A3N[i]*A3D[i]*(rhorel)**(A3D[i]-1)*tau**(A3T[i])

    for i in range(7,51):           # second sum
        phi_res_r = phi_res_r + A3N[i]*np.exp(-(rhorel**A3C[i]))*(rhorel)**(A3D[i]-1)*tau**(A3T[i])*(A3D[i]-A3C[i]*rhorel**(A3C[i]))

    for i in range(3):             # third sum
        phi_res_r = (phi_res_r + A4_real[0,i]*rhorel**(A4_int[0,i])*tau**(A4_int[1,i])*np.exp((-1)*(A4_int[2,i]*(rhorel-A4_int[4,i])**2
                     +A4_int[3,i]*(tau-A4_real[1,i])**2))*(A4_int[0,i]/rhorel-2*A4_int[2,i]*(rhorel-A4_int[4,i])))

    for i in range(2):             # fourth sum
      Theta = (1-tau) + A5_real[4,i]*((rhorel-1)**2)**(1/(2*A5_real[5,i]))
      Delta = (Theta)**2 + A5_real[2,i]*((rhorel-1)**2)**(A5_real[0,i])
      Delta_r = (rhorel - 1)*(A5_real[4,i]*Theta*2/A5_real[5,i]*((rhorel-1)**2)**(1/(2*A5_real[5,1])-1)+
                              2*A5_real[2,i]*A5_real[0,i]*((rhorel-1)**2)**(A5_real[0,i]-1))
      psi = np.exp(-A5_int[0,i]*(rhorel-1)**2-A5_int[1,i]*(tau-1)**2)
      psi_r = psi*(-2*(A5_int[0,i]*(rhorel-1)))

      psi = np.exp(-A5_int[0,i]*(rhorel-1)**2-A5_int[1,i]*(tau-1)**2)
      psi_r = psi*(-2*(A5_int[0,i]*(rhorel-1)))
      phi_res_r = phi_res_r +A5_real[3,i]*(Delta**(A5_real[1,i])*(psi+rhorel*psi_r)+A5_real[1,i]*Delta**(A5_real[1,i]-1)*Delta_r*rhorel*psi)


    #####################  phi_res_rr #######################
    phi_res_rr = 0

    for i in range(7):              # first sum in formula 6
        phi_res_rr = phi_res_rr + A3N[i]*A3D[i]*(rhorel)**(A3D[i]-2)*tau**(A3T[i])*(A3D[i]-1)

    for i in range(7,51):           # second sum
        phi_res_rr = (phi_res_rr + A3N[i]*np.exp((-1)*(rhorel**A3C[i]))*(rhorel)**(A3D[i]-2)*tau**(A3T[i])*((A3D[i]-A3C[i]*rhorel**(A3C[i]))
                     *(A3D[i]-1-A3C[i]*rhorel**(A3C[i]))-A3C[i]**2*rhorel**(A3C[i])))

    for i in range(3):              # third sum
        phi_res_rr = (phi_res_rr + A4_real[0,i] * tau**(A4_int[1,i]) * np.exp( (-1) * (A4_int[2,i] * (rhorel-A4_int[4,i])**2 +
                      A4_int[3,i] * (tau - A4_real[1,i])**2)) * ((-2) * A4_int[2,i] * rhorel**A4_int[0,i] + 4*A4_int[2,i]**2 *
                      rhorel**A4_int[0,i] * (rhorel - A4_int[4,i])**2 - 4 * A4_int[0,i] * A4_int[2,i] * rhorel**(A4_int[0,i]-1)
                      * (rhorel - A4_int[4,i]) + A4_int[0,i] * (A4_int[0,i] - 1) * rhorel**(A4_int[0,i] - 2)))

    for i in range(2):              # fourth sum
        Theta = (1-tau) + A5_real[4,i]*((rhorel-1)**2)**(1/(2*A5_real[5,i]))
        Delta = (Theta)**2 + A5_real[2,i]*((rhorel-1)**2)**(A5_real[0,i])
        Delta_r = (rhorel - 1)*(A5_real[4,i]*Theta*2/A5_real[5,i]*((rhorel-1)**2)**(1/(2*A5_real[5,1])-1)+
                              2*A5_real[2,i]*A5_real[0,i]*((rhorel-1)**2)**(A5_real[0,i]-1))
        Delta_rr = 1 / (rhorel - 1) * Delta_r + (rhorel - 1)**2 * (4 * A5_real[2,i] * A5_real[0,i] * (A5_real[0,i] - 1)
                   *( ( rhorel - 1 )**2 )**(A5_real[0,i]-2) + 2 * A5_real[4,i]**2 * (1/ A5_real[5,i] )**2
                   *(( ( rhorel - 1 )**2 )**(1/ (2 * A5_real[5,i])-1))**2 + A5_real[4,i] * Theta *4/ A5_real[5,i]
                   * (1/( 2 * A5_real[5,i]) - 1)*((rhorel - 1)**2 )**(1/(2*A5_real[5,i])-2 ))
        DeltaB_r = A5_real[1,i]*Delta**(A5_real[1,i]-1)*Delta_r
        DeltaB_rr = A5_real[1,i]*(Delta**(A5_real[1,i]-1)*Delta_rr+(A5_real[1,i]-1)*Delta**(A5_real[1,i]-2)*Delta_r**2)
        psi = np.exp(-A5_int[0,i]*(rhorel-1)**2-A5_int[1,i]*(tau-1)**2)
        psi_r = psi*((-2)*(A5_int[0,i]*(rhorel-1)))
        psi_rr = (2 * A5_int[0,i] * (rhorel - 1)**2 - 1) * 2 * A5_int[0,i] * psi

        phi_res_rr = phi_res_rr + A5_real[3,i]*(Delta**A5_real[1,i]*(2*psi_r+rhorel*psi_rr) + 2*DeltaB_r*(psi+rhorel*psi_r)+DeltaB_rr*rhorel*psi)

        ###################  phi_res_tau ###################

    phi_res_tau = 0
    for i in range(7):              # first sum in formula 6
        phi_res_tau = phi_res_tau + A3N[i]*(rhorel)**(A3D[i])*tau**(A3T[i]-1)*A3T[i]

    for i in range(7,51):           # second sum
        phi_res_tau = phi_res_tau + A3N[i]*(rhorel)**(A3D[i])*tau**(A3T[i]-1)*np.exp(-(rhorel**A3C[i]))*A3T[i]

    for i in range(3):              # third sum !
        phi_res_tau = (phi_res_tau + A4_real[0,i] * rhorel**(A4_int[0,i]) * tau**(A4_int[1,i])
                       * np.exp(-(A4_int[2,i]) * (rhorel-A4_int[4,i])**2 - A4_int[3,i] * (tau - A4_real[1,i])**2 )
                       * ((A4_int[1,i]/tau - 2*A4_int[3,i] * (tau-A4_real[1,i]))) )

    for i in range(2):              # fourth sum
        psi = np.exp(-A5_int[0,i]*(rhorel-1)**2-A5_int[1,i]*(tau-1)**2)
        psi_tau =  (-2)* A4_int[1,i]*(tau-1)*psi
        phi_res_tau = phi_res_tau + A5_real[3,i]*rhorel*(A5_real[1,i]*(-2)*Theta * Delta**(A5_real[1,i]-1)*psi + Delta**A5_real[1,i]*psi_tau)


    ########################## phi_res_tt ##########################

    phi_res_tt = 0
    for i in range(7):              # first sum in formula 6
        phi_res_tt = phi_res_tt + A3N[i]*(rhorel)**(A3D[i])*tau**(A3T[i]-2)*A3T[i]*(A3T[i]-1)

    for i in range(7,51):           # second sum
        phi_res_tt = phi_res_tt + A3N[i] * (rhorel)**(A3D[i]) * tau**(A3T[i]-2) * np.exp(-(rhorel**A3C[i])) *A3T[i]*(A3T[i]-1)

    for i in range(3):              # third sum !
        phi_res_tt = ( phi_res_tt + A4_real[0,i] * rhorel**(A4_int[0,i]) * tau**(A4_int[1,i]) * np.exp((-1)*A4_int[2,i]
                     * (rhorel - A4_int[4,i])**2 - A4_int[3,i] * (tau-A4_real[1,i])**2 )
                     * ( A4_int[1,i]/tau - 2 * A4_int[3,i] * (tau-A4_real[1,i])**2 - (A4_int[1,i]/tau**2 - 2*A4_int[3,i])) )

    for i in range(2):              # fourth sum
        Delta = (Theta)**2 + A5_real[2,i]*((rhorel-1)**2)**(A5_real[0,i])
        DeltaB_tt = (2 * A5_real[1,i] * Delta**(A5_real[1,i]-1) + 4* Theta**2 * A5_real[1,i] * (A5_real[1,i] - 1) * Delta**(A5_real[1,i]-2))
        psi = np.exp(-A5_int[0,i] * (rhorel-1)**2 - A5_int[1,i] * (tau-1)**2)
        psi_tau = (-2)* A5_int[1,i]* (tau - 1) * psi
        DeltaB_t= (-2) * Theta * A5_real[1,i] * Delta**(A5_real[1,i]-1)
        psi_tt= (2 * A5_int[1,i] * (tau-1)**2 - 1 ) * 2 * A5_int[1,i] * psi

        phi_res_tt = phi_res_tt + A5_real[3,i] * rhorel * (DeltaB_tt * psi + 2 * DeltaB_t*psi_tau +Delta**A5_real[1,i]*psi_tt)


    ############################# phi_res_rt #############################

    phi_res_rt = 0
    for i in range(7):              # first sum in formula 6
        phi_res_rt = phi_res_rt + A3N[i]*(rhorel)**(A3D[i]-1)*tau**(A3T[i]-1)*A3D[i]*A3T[i]

    for i in range(7,51):           # second sum
        phi_res_rt = phi_res_rt + A3N[i]*(rhorel)**(A3D[i]-1)*tau**(A3T[i]-1)*A3T[i] * (A3D[i]-A3C[i]*rhorel**A3C[i])*np.exp(-(rhorel**A3C[i]))

    for i in range(3):              # third sum !
        phi_res_rt = ( phi_res_rt + A4_real[0,i] * rhorel**(A4_int[0,i]) * tau**(A4_int[1,i]) * np.exp(-(A4_int[2,i])
                     * (rhorel - A4_int[4,i])**2 - A4_int[3,i] * (tau-A4_real[1,i])**2) * (A4_int[0,i]/rhorel - 2 * A4_int[2,i]
                     * (rhorel - A4_int[4,i])) * (A4_int[1,i]/tau - 2*A4_int[3,i] * (tau - A4_real[1,i])) )

    for i in range(2):              # fourth sum
        Delta = (Theta)**2 + A5_real[2,i]*((rhorel-1)**2)**(A5_real[0,i])
        Delta_r = ( (rhorel-1)*(A5_real[4,i]*Theta*2/A5_real[5,i]*((rhorel-1)**2)**(1/(2*A5_real[5,i])-1)
                   + 2*A5_real[2,i]*A5_real[0,i]*((rhorel-1)**2)**(A5_real[0,i]-1)) )
        psi = np.exp(-A5_int[0,i]*(rhorel-1)**2-A5_int[1,i]*(tau-1)**2)
        psi_tau = (-2)* A4_int[1,i] * (tau-1)*psi
        psi_r = psi*((-2)*(A5_int[0,i]*(rhorel-1)))
        psi_rr = (2 * A5_int[0,i] * (rhorel-1)**2 - 1) * 2 * A5_int[0,i] * psi
        psi_rt = 4 * A5_int[0,i]*A5_int[1,i]*(rhorel-1)*(tau-1)*psi

        DeltaB_r = A5_real[1,i]*Delta**(A5_real[1,i]-1)*Delta_r
        DeltaB_rt = (-A5_real[4,i]*A5_real[1,i]*2/A5_real[5,i]*Delta**(A5_real[1,i]-1)  # CHECK: A5_real(i,2) or A5_real(2,i) in CHIC correct? -> seems to have no effect on Cp
                    *(rhorel-1)*((rhorel-1)**2)**(1/(2*A5_real[5,i])-1) - 2*Theta*A5_real[1,i]*(A5_real[1,i]-1)*Delta**(A5_real[1,i]-2)*Delta_r)
        DeltaB_t=(-2) * Theta * A5_real[1,i] * Delta**(A5_real[1,i]-1)

        phi_res_rt = phi_res_rt + A5_real[3,i]*(Delta**A5_real[1,i]*(psi_tau+rhorel*psi_rt) + rhorel*DeltaB_r*psi_tau
                                                + DeltaB_t*(psi+rhorel*psi_r) + DeltaB_rt*rhorel*psi )

    if (debug >= 4):
        print("phi_res_r = ",phi_res_r)
        print("phi_res_rr = ", phi_res_rr)
        print("phi_res_tau = ", phi_res_tau)
        print("phi_res_tt = ", phi_res_tt)
        print("phi_res_rt = ", phi_res_rt) # write values to type free energy

    # write derivates to type free Energy He
    He = {}
    He['phi']=phi_ideal+phi_res
    He['phi_t']= phi_ideal_t + phi_res_tau
    He['phi_r']=phi_ideal_r+phi_res_r
    He['phi_ideal_r'] = phi_ideal_r
    He['phi_ideal_rr'] = phi_ideal_rr
    He['phi_ideal_t'] = phi_ideal_t
    He['phi_ideal_tt']= phi_ideal_tt
    He['phi_ideal_rt']= phi_ideal_rt
    He['phi_res_r'] =phi_res_r
    He['phi_res_rr'] =phi_res_rr
    He['phi_res_tau']=phi_res_tau
    He['phi_res_tt']=phi_res_tt
    He['phi_res_rt']=phi_res_rt

    return He

#################################################################################################################

# ToDO: calculate from IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic     Properties of Water and Steam
def getrho(p): #valid to 6 GPa,  p_Pa in rho kg*m**2
    # input: p - pressure in Pa
    # return: rho - density in kg/m^3

    p_MPa = p/1e6
    if (p_MPa < 2000):
        rho=8.08910993520484E-08 * p_MPa**3 - 0.000312577913266964*p_MPa**2 + 0.462227514662767*p_MPa + 1003.93618970919
    else:
        rho=193.339237825028 * np.log(p_MPa) - 153.945623678046

    return rho

def getmu(rho,T,debug=0): #http://www.iapws.org/relguide/visc.pdf
    # input:
    #   rho  - rho kg/m**3
    #   T    - Temperature in K
    # return:
    #   mu
    if (debug >= 2):

      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print("++          Part 2: Determine Viscosity of water            ++")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    R = 461.51805 # J/kg K
    # dimensionsless variables
    rhoref = 322.0 #kg/m^3
    rhorel = rho/rhoref
    Tref = 647.096 #K
    Trel = T/Tref
    pref  =22064000. #Pa
    muref = 1.00E-6 #Pa*s

    #numeric values critical-region constants
    xmu = 0.068
    qc = 1/(1.9E-9)
    qd = 1/(1.1E-9)
    v  = 0.630
    gam= 1.239
    e0 = 0.12E-9
    gam0 = 0.06
    Tr = 1.5

    He = GetHelmholtz(rho,T,debug)
    p = (1+He['phi_res_r']*rhorel)*rho*R*T
    prel = p/pref

    ################
    # First factor #
    ################

    A6 = [1.67752,2.20462,0.6366564,-0.241605]
    mu0 = A6[0]+A6[1]/Trel**1+A6[2]/Trel**2+A6[3]/Trel**3
    mu0 = 100 * np.sqrt(Trel)/mu0
    if (debug >= 2):
        print("mu0=",mu0)

    if (debug >= 2):
      print("Trel=",Trel)
      print("rhorel=",rhorel)

    #################
    # Second factor #
    #################
    # i from table 2,

    A8i = [0,1,2,3,0,1,2,3,5,0,1,2,3,4,0,1,0,3,4,3,5]
    A9j = [0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,4,4,5,6,6]
    A7 = [0.520094,0.0850895,-1.08374,-0.289555,0.222531,0.999115,1.88797,1.26613,
          0.120573,-0.281378,-0.906851,-0.772479,-0.489837,-0.25704,0.161913,0.257399,
          -0.0325372,0.0698452,0.00872102,-0.00435673,-0.000593264]

    mu1 = 0.0
    for i in range(21):
      mu1 = mu1 + A7[i]*(1/Trel - 1)**(A8i[i])*(rhorel-1)**(A9j[i])

    mu1 = np.exp(rhorel*mu1)
    if (debug >= 2):
        print("mu1_rel=",mu1)


    ########################
    # Critical enhancement #
    ########################

    # for critical enhancement we need drhorel/dprel
    # numeric derivation

    rho_l =rho*0.99
    rho_r =rho*1.01
    rhorel_l=rho_l/rhoref
    rhorel_r=rho_r/rhoref

    HE = GetHelmholtz(rho_l,T,debug)
    p_l = (1+HE['phi_res_r']*rhorel_l)*rho_l*R*T

    HE = GetHelmholtz(rho_r,T,debug)
    p_r = (1+HE['phi_res_r']*rhorel_r)*rho_r*R*T

    zeta1 = (rho_r-rho_l)/(p_r-p_l)*pref/rhoref


    HE = GetHelmholtz(rho_l,Tr*Tref,debug)
    p_l = (1+HE['phi_res_r']*rhorel_l)*rho_l*R*Tr*Tref

    HE = GetHelmholtz(rho_r,Tr*Tref,debug)
    p_r = (1+HE['phi_res_r']*rhorel_r)*rho_r*R*Tr*Tref

    zeta2 = (rho_r-rho_l)/(p_r-p_l)*pref/rhoref

    dx = rhorel*(zeta1-zeta2*Tr/Trel)

    if (dx >= 0):
      xi = e0* (dx/gam0)**(v/gam)
    else:
      xi = 0

    psiD = np.acos(1/np.sqrt(1 + qd**2*xi*2))
    w = np.sqrt(abs((qc*xi-1)/(qc*xi+1)))*np.tan(psiD/2)
    if (qc*xi > 1):
      L = np.log((1+w)/(1-w))
    else:
      L = 2*np.atan(abs(w))

    if (0 <= xi) & (xi <= 0.3817016416E-9):
      Y = qc/5*xi*(qd*xi)**5*(1-qc*xi+(qc*xi)**2-765/504*(qd*xi)**2)
    elif (xi > 0.3817016416E-9):
      Y = 1/12*np.sin(3*psiD) - np.sin(2*psiD)/(4*qc*xi) + (1-5/4*(qc*xi)**2)*np.sin(psiD)/(qc*xi)**2
      Y = Y - 1/(qc*xi)**3*((1 - 3/2*(qc*xi)**2)*psiD - (abs((qc*xi)**2)-1)**(3/2)*L)
    else:
      Y = 1.0

    mu2 = np.exp(xmu * Y)

    if (debug >= 2):
        print("mu2_rel=",mu2)

    return mu0*mu1*mu2*muref # mu_rel*muref= mu

