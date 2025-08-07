'''
# Author: Lena Noack
# GNU General Public License v3.0 (GPL-3.0)

This module contains a subroutine (getWater) to calculate  thermodynamic properties of water from a Temperature and Pressure.
- It uses the IAPWS-95 formulation (http://www.iapws.org/relguide/IAPWS-95.html) for the density, the thermal expansion coefficient,
  the heat capacity at constant pressure and the thermal conductivity.
- Module verifies that water is fluid at p,T.
- Valid at fluid region, except the area where density is below 800kg/m^3.
- From CHIC, (C) Heistracher and Noack
'''
from Help_functions import getphase,getrho,getmu,GetHelmholtz
import EOS_functions
import numpy as np

def getWater(p, T, debug=0):
    '''
    Input:
      p        pressure in Pa
      T        temperature in K
      debug    print statements for debugging
    Return:
      rho      density in kg/m^3
      alpha    Coefficent of thermal expansion
      Cp       specific isobaric heat capacity in J/kg K  (MPEI: kJ/kg K)
      k        thermal conductivity
      error
    '''

    error = 0

    if (T > 2410.0) | (T < 0): # Verify Temperature is in range
      print("TEMPERATURE OUT OF RANGE (1)",p,T)
      error = 3
      return # end function

    if (p < 0): # Verify pressure is in range
      error = 3
      print("Pressure OUT OF RANGE (2)",p,T)
      return # end function

    phase = getphase(T,p,debug)                # verify phase is fluid
    p_GPa = p*1.0e-9
    if (phase > 0): #if not solid
      error = -phase #2
      #print("STOP Water is solid p =",p,"T = ",T," with phase: ",phase)

      if (phase == 6): # ice VI
        cp, rho, alpha, KT, KS, mu, M, V, V0 = EOS_functions.mineral(p_GPa, T, 'iceVI')
        k = (0.046 * p_GPa ** 2 - 0.0751 * p_GPa + 3.9377)  # Chen et al., 2011, for Ice VII
      elif (phase == 7): # ice VII
        cp, rho, alpha, KT, KS, mu, M, V, V0 = EOS_functions.mineral(p_GPa, T, 'iceVII')
        k = (0.046*p_GPa**2 - 0.0751*p_GPa + 3.9377) # Chen et al., 2011, for Ice VII
      elif (phase == 10): # ice X
        cp, rho, alpha, KT, KS, mu, M, V, V0 = EOS_functions.mineral(p_GPa, T, 'iceX')
        k = (0.046*p_GPa**2 - 0.0751*p_GPa + 3.9377) # Chen et al., 2011, for Ice VII

      return rho, alpha, cp, k, error
    elif (phase == 0):
      print("Pressure or Temperature OUT OF RANGE b",p,T)
      error = 3
      return

    if (debug >= 2):
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print("++                      Water module                           ++")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # ToDO: calculate from IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic     Properties of Water and Steam
    rho = getrho(p)                          # first approximation for density

    if (debug >= 2):
        print("p=",p,"   Pa")
        print("T=",T,"   K")
        print("rho=",rho,"before correction")

    rhoref = 322.0 #kg/m^3              ! Reference Values and constants from IAPWS-95
    Tref = 647.096 #K
    pref = 22.064e+6 #Pa
    kref = 1.0e-3 #W/mK                 ! Reference Values from IAPWS-11 Thermal Conductivity
    muref = 1.0e-6 #Pa s                ! Reference Values from IAPWS-08 Viscosity
    R = 461.51805 #J/kg K
    Trel = T/Tref
    tau = Tref/T
    rhorel=rho/rhoref

    if (debug >= 3):
        print("Trel=",Trel,", rhorel=",rhorel)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++         Part 1: Determine thermodynamic properties          ++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #http://twt.mpei.ac.ru/mcs/worksheets/iapws/IAPWS95.xmcd
        print("tau=",tau,", rhorel=",rhorel)

    He = GetHelmholtz(rho,T,debug)    # Get Properties with approximated density

    ###################### Iteration for pressure correction ######################

    # Thermodynamic Properties calculated with aprox. density are used to correct density
    # The density is adjusted until the difference between pressure calculated with Helmholtz equations and input-pressure
    # is less than 10^(-5) * input pressure

    p_new = (1+He['phi_res_r']*rhorel)*rho*R*T          # new pressure is calculated with Helmholtz equations
    j = 0
    if (debug >= 3):
        print("rho- = ",rho, "p_new- =",p_new,"count = ",j)

    stepsize = 100.

    while (abs(p-p_new) > 0.1 * p):           # Break if difference in pressures is less than 10^(-5)
        if (j > 400):                                # Break after 400 steps
          print("Iteration max -Iteration did not converge")
          return
        elif ((p-p_new) < 0):                      # decrease rho until sign changes
          while ((p-p_new) < 0):
            rho = rho - stepsize
            rhorel = rho/rhoref
            He = GetHelmholtz(rho,T,debug)
            p_new = (1+He['phi_res_r']*rhorel)*rho*R*T
            j += 1
            if (debug >= 3):
              print("rho- = ",rho, "p_new- =",p_new,"count = ",j)
        elif (p-p_new > 0):                        # increases rho until sign changes
          while ((p-p_new) > 0):
            rho = rho + stepsize
            rhorel = rho/rhoref
            He = GetHelmholtz(rho,T,debug)
            p_new = (1+He['phi_res_r']*rhorel)*rho*R*T
            j += 1
            if (debug >= 3):
              print("rho+ = ",rho, "p_new+ =",p_new,"count = ",j)

        stepsize = stepsize * 0.3                  # decrease stepsize

        if (rho < 400):
            print("Density to small, ",rho)
            error = 1
            return

    if (debug >= 3):
        print(" FINAL DENSITY SET TO  rho = ", rho)

    if (debug == 10 ): # if debug = 10, sets rho to 838.025 after iteration (benchmark for helmholtz subroutine)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   WARNING debug    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   WARNING     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!   density is set to rho = 838.025  !!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!Benchmark with T = 500K from IAPWS-95 table 6!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   WARNING     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        rho = 838.025
        rhorel=rho/rhoref

    if (debug >= 2):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++                      Water Properties                       ++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if (debug ==3):
        He = GetHelmholtz(rho,T,4)
    else:
        He = GetHelmholtz(rho,T,debug)
    #thermodynamic properties (Table 3)

    #isochoric heat capacity cv IAPWS-95
    cv = (-R)* tau**2 *(He['phi_ideal_tt'] + He['phi_res_tt'])

    #isobaric heat capacity cp IAPWS-95
    cp =cv + ((1+rhorel*He['phi_res_r'] - rhorel*tau*He['phi_res_rt'])**2)/(1+2*rhorel*He['phi_res_r']+rhorel**2*He['phi_res_rr']) * R

    #speed of sound IAPWS-95
    w = 1 + 2 * rhorel*He['phi_res_r'] + rhorel**2*He['phi_res_rr']
    w = w - ((1 + rhorel*He['phi_res_r']-rhorel*tau*He['phi_res_rt'])**2)/(tau**2*(He['phi_ideal_tt'] + He['phi_res_tt']))
    w = np.sqrt(R*T*w)

    #pressure IAPWS-95
    x = (1 + He['phi_res_r']*rhorel)*rho*R*T

    if (debug >= 2):
        print("Pressure (with Helmholtz) = ", x , "Pa")
        print("Temperatur                 = ", T , "K ")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if (debug >= 4):
        print(" R = ",R)
        print(" T = ",T)
        print("rhorel =  ",rhorel)
        print("HE%phi_res_r ",He['phi_res_r'])

    #coefficient thermal expansion with Feistel et al. Thermodynamic potentials for fluid water, ice and seawater
    alpha = np.sqrt(cp/cv*(cp-cv)/(w**2*T)) #*((alpha2)/ABS(alpha2))
                                                                    # next to if to determine sign of alpha
    if ((T < (277.025401662*np.exp(-7.76352993315771E-10*p))) & (p < 2.75E+007)):
        alpha = -alpha

    mu = getmu(rho,T,debug)         # get viscosity with function getmu
    if (debug >= 2):
        print("T     = ", T , " K")
        print("rho   = ", rho ," kg*m^(-3) (with correction) ")
        print("cv    = ", cv," J/(kg*K)")
        print("cp    = ", cp," J/(kg*K)")
        print("w     = ", w, " m/s")
        print("alpha =  ", alpha, "1/K       (ABS(alpha)) ")
        print("mu    =  " , mu ,"Pa * s")

    if (debug >= 3):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++          Part 3: Determine conductivity of water            ++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # IAPWS Formulation 2011 for the thermal conductivity of Ordinary Water Substance
    # http://twt.mpei.ac.ru/mcs/worksheets/iapws/wspTCPT.xmcd ( online calculation)

    ################
    # First factor #
    ################
    k0 = np.sqrt(Trel)/(2.443221E-3+1.323095E-2/Trel+6.770357E-3/Trel**2-3.454586E-3/Trel**3+4.096266E-4/Trel**4)
    if (debug >= 3):
        print("k0_rel=",k0)

    #################
    # Second factor #
    #################

    #dimension(5,6)
    A1 = np.array([[1.60397357,-0.646013523,0.111443906,0.102997357,-0.0504123634],
          [0.0060985926,2.33771842,-2.78843778,1.53616167,-0.463045512],
          [0.0832827019,-0.0071920125,2.19650529,-4.54580785,3.55777244],
          [-1.40944978,0.275418278,-0.0205938816,-1.21051378,1.60812989],
          [-0.621178141,0.0716373224,0.,0.,-2.720337],
          [4.57586331,-3.18369245,1.1168348,-0.19268305,0.012913842]]).transpose()

    k1 = 0.0
    for i in range(5):
      for j in range(6):
        k1 = k1 + (1.0/Trel-1.0)**i * A1[i,j]*(rhorel-1.0)**j

    # print(k1,np.exp(rhorel*k1))
    if k1<100: # otherwise anyway wrong k1 value
        k1 = np.exp(rhorel*k1)
    if (debug >= 3):
        print("k1_rel=",k1)

    ########################
    # Critical enhancement #
    ########################

    # Reference Values and Constants                                                                   !ToDo check units
    lambdav = 177.8514
    qd = 1/(0.401E-9)
    v  = 0.630
    gam= 1.239
    e0 = 0.13E-9
    Gam0 = 0.06
    Tr = 1.5
    kref = 1E-3

    ####################### Numeric derivation rho to p #######################

    rho_l = rho*0.99
    rho_r = rho*1.01
    rhorel_l = rho_l/rhoref
    rhorel_r = rho_r/rhoref
    #TrRef = Tr *Tr

    He = GetHelmholtz(rho_l,T,0)
    p_l = (1+He['phi_res_r']*rhorel_l)*rho_l*R*T

    He = GetHelmholtz(rho_r,T,0)
    p_r = (1+He['phi_res_r']*rhorel_r)*rho_r*R*T

    zeta1 = (rho_r-rho_l)/(p_r-p_l)*pref/rhoref

    #rho_l =rho*0.99
    He = GetHelmholtz(rho_l,(Tr*Tref),0)
    p_l = (1+He['phi_res_r']*rhorel_l)*rho_l*R*Tr*Tref
    #rho_r =rho*1.01
    He = GetHelmholtz(rho_r,(Tr*Tref),0)
    p_r = (1+He['phi_res_r']*rhorel_r)*rho_r*R*Tr*Tref

    zeta2 = (rho_r-rho_l)/(p_r-p_l)*pref/rhoref

    dx = rhorel*(zeta1-zeta2*Tr/Trel)

    if (dx >= 0):
        xi = e0* (dx/Gam0)**(v/gam)
    else:
        xi = 0

    y = qd*xi
    if (y >= 1.2E-7):
        Z = 2/(PI*y)*(((1 - cv/cp)*np.atan(y) + cv/cp*y) - ( 1 - np.exp((-1)/(1/y+y**2/(3*rhorel**2)))))
    else:
        Z = 0

    k2 = lambdav * rhorel*cp*Trel/mu*Z*muref/R
    if (debug >= 3):
        print("k2_rel=",k2)

    # get final k

    if ((k1 > 0.0) & (k1 < 100000.0)):
        k = k0*k1
    else:
        k = k0

    if ((k2 > 0.0) & (k2 < 100000.0)):
        k = (k+k2)*kref
    else:
        k = k*kref

    #k = (k0*k1 + k2)*kref  ! problem, k1 and k2 kan be infinity/NaN for high pressures

    if (debug >= 2):
        print("k =      ", k , "W/K")

    if ((k<1e-20) | (k>1000000.0)):
        k = 0.56

    #101 CpCv = cp/cv


    # Calculation Cp, Cv, mu etc. see
    # http://twt.mpei.ac.ru/mcs/worksheets/iapws/IAPWS95.xmcd
    # http://www.iapws.org/relguide/IAPWS95-Rev.pdf

    return rho, float(alpha), float(cp), float(k), error
