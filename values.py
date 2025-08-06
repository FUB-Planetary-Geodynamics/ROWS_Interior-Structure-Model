import numpy as np
#####################################################
# version based on Matlab code (C) Sonja Zierke, Alexander Balduin, Lena Noack
#####################################################
class values():     
    """EOS properties
    structure for mineral values essential for partition coefficient
    calculation (Verteilungskoeffizient)
    Reference: Mantle - Stixrude and Lithgow-Bertelloni 2011
    Reference: Iron - Bouchet et al. 2013

    F0      =  Helmholtz free energy (kJ/mol)
    V0      =  Volume(cm^3/mol) --> *1000 at the end [mm^3/mol]
    KT0     =  bulk modulus(GPa)
    KTp     =  bulk modulus derivative
    theta0  =  characteristic vibrational temperature(K)
    gamma0  =  Grüneisen parameter
    q0      =  second logarithmic volume derivative / beta in EOS
    G0      =  reference shear modulus(GPa)
    Gp      =  shear modulus derivitive
    eta     =    """
#####################################################

####################################################################################
    def __init__(p, name='fo'):        
####################################################################################

        p.T0 = 298   # reference temperature
        p.EOS = 3    # Birch-Murnaghan 3rd order (both values are overwritten for iron values at the end)
        p.en = p.EOS # energy term EOS

        #Elemental masses
        O  = 15.999
        H  = 1.007825
        C  = 12.011
        S  = 32.065
        Na = 22.98976928
        Mg = 24.305
        Al = 26.9815384
        Si = 28.085
        Ca = 40.078
        Fe = 55.845

        mol = 6.02214076e23 #atoms

        if name=='an':
            # Phase:feldspar(plg) Species:Anorthite(an) Formula:Ca[Al2Si2]O8
            p.name = name;
            p.F0 = -4015;
            p.V0 = 100.61;
            p.KT0 = 84;
            p.KTp = 4;
            p.theta0 = 752;
            p.gamma0 = 0.39;
            p.q0 = 1.00;
            p.G0 = 40;
            p.Gp = 1.1;
            p.eta = 1.60;
            p.M = Ca + Al*2 + Si*2 + O*8;
            p.n = 13; 
        elif name=='ab':
            #Phase:feldspar(plg) Species:Albite(ab) Formula:Na[AlSi3]O8
            p.name = name;
            p.F0 = -3719;
            p.V0 = 100.45;
            p.KT0 = 60;
            p.KTp = 4;
            p.theta0 = 716;
            p.gamma0 = 0.57;
            p.q0 = 1.00;
            p.G0 = 36;
            p.Gp = 1.4;
            p.eta = 1.00;
            p.M = Na + Al + Si*3 + O*8;
            p.n = 13;
        elif name=='sp':
            #Phase:spinel(sp) Species:Spinel(sp) Formula:(Mg3Al)(Al7Mg)Ol6
            p.name = name;
            p.F0 = -8668;
            p.V0 = 159.05;
            p.KT0 = 197;
            p.KTp = 5.7;
            p.theta0 = 843;
            p.gamma0 = 1.02;
            p.q0 = 2.70;
            p.G0 = 108;
            p.Gp = 0.4;
            p.eta = 2.70;
            p.M = Mg*4 + Al*8 + O*16;
            p.n = 28;
        elif name=='hc':
            #Phase:spinel(sp) Species:Hercynite(hc) Formula:(Fe3Al)(Al7Fe)Ol6
            p.name = name;
            p.F0 = -7324;
            p.V0 = 163.37;
            p.KT0 = 209;
            p.KTp = 5.7;
            p.theta0 = 763;
            p.gamma0 = 1.22;
            p.q0 = 2.70;
            p.G0 = 84;
            p.Gp = 0.4;
            p.eta = 2.80;
            p.M = Fe*4 + Al*8 + O*16;
            p.n = 28;
        elif name=='fo':
            #Phase:olivine(ol) Species:Forsterite(fo) Formula:Mg2SiO4
            p.name = name;
            p.F0 = -2055;
            p.V0 = 43.6;
            p.KT0 = 128;
            p.KTp = 4.2;
            p.theta0 = 809;
            p.gamma0 = 0.99;
            p.q0 = 2.10;
            p.G0 = 82;
            p.Gp = 1.5;
            p.eta = 2.30;
            p.M = Mg*2 + Si + O*4;
            p.n = 7;
        elif name=='fa':
            #Phase:olivine(ol) Species:Fayalite(fa) Formula:Fe2SiO4
            p.name = name;
            p.F0 = -1371;
            p.V0 = 46.29;
            p.KT0 = 135;
            p.KTp = 4.2;
            p.theta0 = 619;
            p.gamma0 = 1.06;
            p.q0 = 3.60;
            p.G0 = 51;
            p.Gp = 1.5;
            p.eta = 1.00;
            p.M = Fe*2 + Si + O*4;
            p.n = 7;
        elif name=='mgwa':
            #Phase:wadsleyite(wa) Species:Mg-Wadsleyite(mgwa) Formula:Mg2SiO4
            p.name = name;
            p.F0 = -2028;
            p.V0 = 40.52;
            p.KT0 = 169;
            p.KTp = 4.3;
            p.theta0 = 844;
            p.gamma0 = 1.21;
            p.q0 = 2.00;
            p.G0 = 112;
            p.Gp = 1.4;
            p.eta = 2.60;
            p.M = Mg*2 + Si + O*4;
            p.n = 7;
        elif name=='fewa':
            #Phase:wadsleyite(wa) Species:Fe-Wadsleyite(fewa) Formula:Fe2SiO4
            p.name = name;
            p.F0 = -1365;
            p.V0 = 42.8;
            p.KT0 = 169;
            p.KTp = 4.3;
            p.theta0 = 665;
            p.gamma0 = 1.21;
            p.q0 = 2.00;
            p.G0 = 72;
            p.Gp = 1.4;
            p.eta = 1.00;
            p.M = Fe*2 + Si + O*4;
            p.n = 7;
        elif name=='mgri':
            #Phase:ringwoodite(ri) Species:Mg-Ringwoodite(mgri) Formula:Mg2SiO4
            p.name = name;
            p.F0 = -2017;
            p.V0 = 39.49;
            p.KT0 = 185;
            p.KTp = 4.2;
            p.theta0 = 878;
            p.gamma0 = 1.11;
            p.q0 = 2.40;
            p.G0 = 123;
            p.Gp = 1.4;
            p.eta = 2.30;
            p.M = Mg*2 + Si + O*4;
            p.n = 7;
        elif name=='feri':
            #Phase:ringwoodite(ri) Species:Fe-Ringwoodite(feri) Formula:Fe2SiO4
            p.name = name;
            p.F0 = -1363;
            p.V0 = 41.86;
            p.KT0 = 213;
            p.KTp = 4.2;
            p.theta0 = 679;
            p.gamma0 = 1.27;
            p.q0 = 2.40;
            p.G0 = 92;
            p.Gp = 1.4;
            p.eta = 1.80;
            p.M = Fe*2 + Si + O*4;
            p.n = 7;
        elif name=='en':
            #Phase:orthopyroxene(opx) Species:Enstatite(en) Formula:MgMgSi2O6
            p.name = name;
            p.F0 = -2913;
            p.V0 = 62.68;
            p.KT0 = 107;
            p.KTp = 7;
            p.theta0 = 812;
            p.gamma0 = 0.78;
            p.q0 = 3.40;
            p.G0 = 77;
            p.Gp = 1.5;
            p.eta = 2.50;
            p.M = Mg*2 + Si*2 + O*6;
            p.n = 7;
        elif name=='fs':
            #Phase:orthopyroxene(opx) Species:Ferrosilite(fs) Formula:FeFeSi2O6
            p.name = name;
            p.F0 = -2226;
            p.V0 = 65.94;
            p.KT0 = 101;
            p.KTp = 7;
            p.theta0 = 674;
            p.gamma0 = 0.72;
            p.q0 = 3.40;
            p.G0 = 52;
            p.Gp = 1.5;
            p.eta = 1.10;
            p.M = Fe*2 + Si*2 + O*6;
            p.n = 10;
        elif name=='mgts':
            #Phase:orthopyroxene(opx) Species:Mg-Tschermalcs(mgts) Formula:MgAl[SiAl]O6
            p.name = name;
            p.F0 = -3003;
            p.V0 = 59.14;
            p.KT0 = 107;
            p.KTp = 7;
            p.theta0 = 784;
            p.gamma0 = 0.78;
            p.q0 = 3.40;
            p.G0 = 97;
            p.Gp = 1.5;
            p.eta = 2.50;
            p.M = Mg + Al*2 + Si + O*6;
            p.n = 10;
        elif name=='odi':
            #Phase:orthopyroxene(opx) Species:Ortho-Diopside(odi) Formula:CaMgSi2O6
            p.name = name;
            p.F0 = -3016;
            p.V0 = 68.05;
            p.KT0 = 107;
            p.KTp = 7;
            p.theta0 = 745;
            p.gamma0 = 0.78;
            p.q0 = 3.40;
            p.G0 = 60;
            p.Gp = 1.5;
            p.eta = 1.40;
            p.M = Ca + Mg + Si*2 + O*6;
            p.n = 10;
        elif name=='di':
            #Phase:clinopyroxene(opx) Species:Diopside(di) Formula:CaMgSi2O6
            p.name = name;
            p.F0 = -3030;
            p.V0 = 66.04;
            p.KT0 = 112;
            p.KTp = 5.2;
            p.theta0 = 782;
            p.gamma0 = 0.96;
            p.q0 = 1.50;
            p.G0 = 67;
            p.Gp = 1.4;
            p.eta = 1.60;
            p.M = Ca + Mg + Si*2 + O*6;
            p.n = 10;    
        elif name=='he':
            #Phase:clinopyroxene(cpx) Species:Hedenbergite(he) Formula:CaFeSi2O6
            p.name = name;
            p.F0 = -2677;
            p.V0 = 67.87;
            p.KT0 = 119;
            p.KTp = 5.2;
            p.theta0 = 702;
            p.gamma0 = 0.94;
            p.q0 = 1.50;
            p.G0 = 61;
            p.Gp = 1.2;
            p.eta = 1.60;
            p.M = Ca + Fe + Si*2 + O*6;
            p.n = 10;
        elif name=='cen':
            #Phase:clinopyroxene Species:Clinoenstatite(cen) Formula:MgMgSi2O6
            p.name = name;
            p.F0 = -2906;
            p.V0 = 62.5;
            p.KT0 = 112;
            p.KTp = 5.2;
            p.theta0 = 805;
            p.gamma0 = 0.96;
            p.q0 = 1.50;
            p.G0 = 81;
            p.Gp = 1.7;
            p.eta = 1.7;
            p.M = Mg*2 + Si*2 + O*6;
            p.n = 10;
        elif name=='cats':
            #Phase:clinopyroxene Species:Ca-Tschermaks(cats) Formula:CaAl(SiAl)O6
            p.name = name;
            p.F0 = -3120;
            p.V0 = 63.57;
            p.KT0 = 112;
            p.KTp = 5.2;
            p.theta0 = 804;
            p.gamma0 = 0.78;
            p.q0 = 1.5;
            p.G0 = 76;
            p.Gp = 1.6;
            p.eta = 2.0;
            p.M = Ca + Al*2 + Si + O*6;
            p.n = 10;
        elif name=='jd':
            #Phase:clinopyroxene Species:Jadeite(jd) Formula:NaAlSi2O6
            p.name = name;
            p.F0 = -2855;
            p.V0 = 60.51;
            p.KT0 = 142;
            p.KTp = 5.2;
            p.theta0 = 821;
            p.gamma0 = 0.90;
            p.q0 = 0.4;
            p.G0 = 85;
            p.Gp = 1.4;
            p.eta = 2.2;
            p.M = Na + Al + Si*2 + O*6;
            p.n = 10;
        elif name=='hpcen':
            #Phase:HP-clinopyroxene Species:HP-Clinoenstatite(hpcen) Formula:Mg2Si2O6
            p.name = name;
            p.F0 = -2905;
            p.V0 = 60.76;
            p.KT0 = 116;
            p.KTp = 6.2;
            p.theta0 = 824;
            p.gamma0 = 1.12;
            p.q0 = 0.20;
            p.G0 = 88;
            p.Gp = 1.8;
            p.eta = 2.10;
            p.M = Mg*2 + Si*2 + O*6;
            p.n = 10;
        elif name=='hpcfs':
            #Phase:HP-clinopyroxene Species:HP-Clinoferrosilite(hpcfs)  Formula:Fe2Si2O6
            p.name = name;
            p.F0 = -2222;
            p.V0 = 63.85;
            p.KT0 = 116;
            p.KTp = 6.2;
            p.theta0 = 692;
            p.gamma0 = 1.12;
            p.q0 = 0.20;
            p.G0 = 71;
            p.Gp = 1.8;
            p.eta = 0.80;
            p.M = Fe*2 + Si*2 + O*6;
            p.n = 10;
        elif name=='capv':
            #Phase:Ca-perovskite Species:Ca-Perovskite(capv) Formula:CaSiO3
            p.name = name;
            p.F0 = -1463;
            p.V0 = 27.45;
            p.KT0 = 236;
            p.KTp = 3.9;
            p.theta0 = 796;
            p.gamma0 = 1.89;
            p.q0 = 0.90;
            p.G0 = 157;
            p.Gp = 2.2;
            p.eta = 1.30;
            p.M = Ca + Si + O*3;
            p.n = 5;
        elif name=='mgak':
            #Phase:akimotoite(ak) Species:Mg-Akimotoite(mgak) Formula:MgSiO3
            p.name = name;
            p.F0 = -1410;
            p.V0 = 26.35;
            p.KT0 = 211;
            p.KTp = 5.6;
            p.theta0 = 934;
            p.gamma0 = 1.19;
            p.q0 = 2.30;
            p.G0 = 132;
            p.Gp = 1.6;
            p.eta = 2.80;
            p.M = Mg + Si + O*3;
            p.n = 5;
        elif name=='feak':
            #Phase:akimotoite Species:Fe-Akimotoite(feak) Formula:FeSiO3
            p.name = name;
            p.F0 = -1068;
            p.V0 = 26.85;
            p.KT0 = 211;
            p.KTp = 5.6;
            p.theta0 = 888;
            p.gamma0 = 1.19;
            p.q0 = 2.30;
            p.G0 = 150;
            p.Gp = 1.6;
            p.eta = 3.50;
            p.M = Fe + Si + O*3;
            p.n = 5;
        elif name=='co':
            #Phase:akimotoite Species:Corundum(co) Formula:AlAlO3
            p.name = name;
            p.F0 = -1582;
            p.V0 = 25.58;
            p.KT0 = 253;
            p.KTp = 4.3;
            p.theta0 = 933;
            p.gamma0 = 1.32;
            p.q0 = 1.30;
            p.G0 = 163;
            p.Gp = 1.6;
            p.eta = 2.80;
            p.M = Al*2 + O*3;
            p.n = 5;
        elif name=='py':
            #Phase:garnet(gt.mj) Species:Pyrope(py) Formula:Mg3AlAlSi3Ol2
            p.name = name;
            p.F0 = -5936;
            p.V0 = 113.08;
            p.KT0 = 170;
            p.KTp = 4.1;
            p.theta0 = 823;
            p.gamma0 = 1.01;
            p.q0 = 1.40;
            p.G0 = 94;
            p.Gp = 1.4;
            p.eta = 1.00;
            p.M = Mg*3 + Al*2 + Si*3 + O*12;
            p.n = 20;
        elif name=='al':
            #Phase:garnet Species:Almandine(al) Formula:Fe3AlAlSi3Ol2
            p.name = name;
            p.F0 = -4935;
            p.V0 = 115.43;
            p.KT0 = 174;
            p.KTp = 4.9;
            p.theta0 = 741;
            p.gamma0 = 1.06;
            p.q0 = 1.40;
            p.G0 = 96;
            p.Gp = 1.4;
            p.eta = 2.10;
            p.M = Fe*3 + Al*2 + Si*3 + O*12;
            p.n = 20;
        elif name=='gr':
            #Phase:garnet Species:Grossular(gr) Formula:Ca3AIAlSi3Ol2
            p.name = name;
            p.F0 = -6278;
            p.V0 = 125.12;
            p.KT0 = 167;
            p.KTp = 3.9;
            p.theta0 = 823;
            p.gamma0 = 1.05;
            p.q0 = 1.90;
            p.G0 = 109;
            p.Gp = 1.2;
            p.eta = 2.40;
            p.M = Ca*3 + Al*2 + Si*3 + O*12;
            p.n = 20;
        elif name=='mgmj':
            #Phase:garnet Species:Mg-Majorite(mgmj) Formula:Mg3MgSiSi3Ol2
            p.name = name;
            p.F0 = -5691;
            p.V0 = 114.32;
            p.KT0 = 165;
            p.KTp = 4.2;
            p.theta0 = 822;
            p.gamma0 = 0.98;
            p.q0 = 1.50;
            p.G0 = 85;
            p.Gp = 1.4;
            p.eta = 1.00;
            p.M = Mg*4 + Si*4 + O*12;
            p.n = 20;
        elif name=='jdmj':
            #Phase:garnet Species:Jd-Majorite(jdmj) Formula:(Na2Al)AlSiSi3Ol2
            p.name = name;
            p.F0 = -5519;
            p.V0 = 110.94;
            p.KT0 = 177;
            p.KTp = 4.1;
            p.theta0 = 896;
            p.gamma0 = 1.01;
            p.q0 = 1.40;
            p.G0 = 125;
            p.Gp = 1.4;
            p.eta = 3.30;
            p.M = Na*2 + Al*2 + Si*4 + O*12;
            p.n = 20;
        elif name=='qtz':
            #Phase:quartz(qtz) Species:Quartz(qtz) Formula:SiO2
            p.name = name;
            p.F0 = -859;
            p.V0 = 23.67;
            p.KT0 = 50;
            p.KTp = 4.3;
            p.theta0 = 816;
            p.gamma0 = 0.00;
            p.q0 = 1.00;
            p.G0 = 45;
            p.Gp = 1;
            p.eta = 2.40;
            p.M = Si + O*2;
            p.n = 3;
        elif name=='coes':
            #Phase:coesite(coes) Species:Coesite(coes) Formula:SiO2
            p.name = name;
            p.F0 = -855;
            p.V0 = 20.66;
            p.KT0 = 114;
            p.KTp = 4;
            p.theta0 = 857;
            p.gamma0 = 0.39;
            p.q0 = 1.00;
            p.G0 = 62;
            p.Gp = 1.2;
            p.eta = 2.40;
            p.M = Si + O*2;
            p.n = 3;
        elif name=='st':
            #Phase:stishovite(st) Species:Stishovite(st) Formula:SiO2
            p.name = name;
            p.F0 = -819;
            p.V0 = 14.02;
            p.KT0 = 314;
            p.KTp = 3.8;
            p.theta0 = 1108;
            p.gamma0 = 1.37;
            p.q0 = 2.80;
            p.G0 = 220;
            p.Gp = 1.9;
            p.eta = 4.60;
            p.M = Si + O*2;
            p.n = 3;
        elif name=='seif':
            #Phase:seifertite(seif) Species:Seifertite(seif) Formula:SiO2
            p.name = name;
            p.F0 = -794;
            p.V0 = 13.67;
            p.KT0 = 328;
            p.KTp = 4;
            p.theta0 = 1141;
            p.gamma0 = 1.37;
            p.q0 = 2.80;
            p.G0 = 227;
            p.Gp = 1.8;
            p.eta = 5.00;
            p.M = Si + O*2;
            p.n = 3;
        elif name=='mgpv':
            #Phase:perovskite(pv) Species:Mg-Perovskite(mgpv) Formula:MgSiO3
            p.name = name;
            p.F0 = -1368;
            p.V0 = 24.45;
            p.KT0 = 251;
            p.KTp = 4.1;
            p.theta0 = 905;
            p.gamma0 = 1.57;
            p.q0 = 1.10;
            p.G0 = 173;
            p.Gp = 1.7;
            p.eta = 2.60;
            p.M = Mg + Si + O*3;
            p.n = 5;
        elif name=='fepv':
            #Phase:perovskite Species:Fe-Perovskite(fepv) Formula:FeSiO3
            p.name = name;
            p.F0 = -1041;
            p.V0 = 25.49;
            p.KT0 = 272;
            p.KTp = 4.1;
            p.theta0 = 871;
            p.gamma0 = 1.57;
            p.q0 = 1.10;
            p.G0 = 133;
            p.Gp = 1.4;
            p.eta = 2.30;
            p.M = Fe + Si + O*3;
            p.n = 5;
        elif name=='rh2o3':
            #Phase:perovskite Species:Rh2O3-II(rh2o3) Formula:AlAlO3
            p.name = name;
            p.F0 = -1534;
            p.V0 = 24.94;
            p.KT0 = 258;
            p.KTp = 4.1;
            p.theta0 = 886;
            p.gamma0 = 1.57;
            p.q0 = 1.10;
            p.G0 = 171;
            p.Gp = 1.5;
            p.eta = 2.50;
            p.M = Al*2 + O*3;
            p.n = 5;
        elif name=='mppv':
            #Phase:post-perovskite Species:Mg-Post-Perovskite(mppv) Formula:MgSiO3
            p.name = name;
            p.F0 = -1348;
            p.V0 = 24.42;
            p.KT0 = 231;
            p.KTp = 4;
            p.theta0 = 855;
            p.gamma0 = 1.89;
            p.q0 = 1.10;
            p.G0 = 150;
            p.Gp = 2;
            p.eta = 1.20;
            p.M = Mg + Si + O*3;
            p.n = 5;
        elif name=='fppv':
            #Phase:post-perovskite Species:Fe-Post-Perovskite(fpv) Formula:FeSiO3
            p.name = name;
            p.F0 = -982;
            p.V0 = 25.46;
            p.KT0 = 231;
            p.KTp = 4;
            p.theta0 = 782;
            p.gamma0 = 1.89;
            p.q0 = 1.10;
            p.G0 = 129;
            p.Gp = 1.4;
            p.eta = 1.40;
            p.M = Fe + Si + O*3;
            p.n = 5;
        elif name=='appv':
            #Phase:post-perovskite Species:AI-Post-Perovskite(appv) Formula:AlAlO3
            p.name = name;
            p.F0 = -1378;
            p.V0 = 23.85;
            p.KT0 = 249;
            p.KTp = 4;
            p.theta0 = 762;
            p.gamma0 = 1.65;
            p.q0 = 1.10;
            p.G0 = 92;
            p.Gp = 1.8;
            p.eta = 2.80;
            p.M = Al*2 + O*3;
            p.n = 5;
        elif name=='pe':
            #Phase:magnesiowüstite Species:Periclase(pe) Formula:MgO
            p.name = name;
            p.F0 = -569;
            p.V0 = 11.24;
            p.KT0 = 161;
            p.KTp = 3.8;
            p.theta0 = 767;
            p.gamma0 = 1.36;
            p.q0 = 1.70;
            p.G0 = 131;
            p.Gp = 2.1;
            p.eta = 2.80;
            p.M = Mg + O;
            p.n = 2;
        elif name=='wu':
            #Phase:magnesiowüstite Species:Wüstite(wu) Formula:FeO
            p.name = name;
            p.F0 = -242;
            p.V0 = 12.26;
            p.KT0 = 179;
            p.KTp = 4.9;
            p.theta0 = 454;
            p.gamma0 = 1.53;
            p.q0 = 1.70;
            p.G0 = 59;
            p.Gp = 1.4;
            p.eta = -0.10;
            p.M = Fe + O;
            p.n = 2;
        elif name=='mgcf':
            #Phase:Ta-ferrite(cf) Species:Mg-Ca-Ferrite(mgcf) Formula:MgAlAlO4
            p.name = name;
            p.F0 = -2122;
            p.V0 = 36.18;
            p.KT0 = 211;
            p.KTp = 4.1;
            p.theta0 = 838;
            p.gamma0 = 1.31;
            p.q0 = 1.00;
            p.G0 = 130;
            p.Gp = 1.8;
            p.eta = 2.10;
            p.M = Mg + Al*2 + O*4;
            p.n = 7;
        elif name=='fecf':
            #Phase:fa-ferrite Species:lb-Ta-Ferrite(fecf) Formula:FeAlAlO4
            p.name = name;
            p.F0 = -1790;
            p.V0 = 37.26;
            p.KT0 = 211;
            p.KTp = 4.1;
            p.theta0 = 804;
            p.gamma0 = 1.31;
            p.q0 = 1.00;
            p.G0 = 152;
            p.Gp = 1.8;
            p.eta = 3.00;
            p.M = Fe + Al*2 + O*4;
            p.n = 7;
        elif name=='nacf':
            #Phase:Ca-ferrite Species:Na-Ca-Ferrite(nacf) Formula:NaAlSiO4
            p.name = name;
            p.F0 = -1851;
            p.V0 = 36.27;
            p.KT0 = 158;
            p.KTp = 4.3;
            p.theta0 = 812;
            p.gamma0 = 1.17;
            p.q0 = 1.00;
            p.G0 = 121;
            p.Gp = 2.1;
            p.eta = 1.60;
            p.M = Na + Al + Si + O*4;
            p.n = 7;
        elif name=='ky':
            #Phase:kyanite(ky) Species:Kyanite(ky) Formula:Al2SiO5
            p.name = name;
            p.F0 = -2446;
            p.V0 = 44.23;
            p.KT0 = 160;
            p.KTp = 4;
            p.theta0 = 943;
            p.gamma0 = 0.93;
            p.q0 = 1.00;
            p.G0 = 121;
            p.Gp = 1.7;
            p.eta = 3.00;
            p.M = Al*2 + Si + O*5;
            p.n = 8;
        elif name=='neph':
            #Phase:nephclinc(neph) Species:Nephclinc(neph) Formula:NaAlSiO4
            p.name = name;
            p.F0 = -1993;
            p.V0 = 54.67;
            p.KT0 = 53;
            p.KTp = 4;
            p.theta0 = 701;
            p.gamma0 = 0.69;
            p.q0 = 1.00;
            p.G0 = 31;
            p.Gp = 1.3;
            p.eta = 0.60;
            p.M = Na + Al + Si + O*4;
            p.n = 7;
        ####################
        # Iron core phases #
        ####################
        elif name=='fehcp':
            #Phase:iron hcp phase
            p.name = name;
            p.Z = 26; 
            p.T0 = 300.0; # [K]
            p.n = 1;
            p.V0 = 6.29; # [cm^3 mol^-1]
            p.M = Fe; #55.845e6; % Fe [g mol^-1]
            p.KT0 = 253.844; # [GPa]
            p.KTp = 4.719; # [1]
            p.theta0 = 44.574; # [K]
            p.gamma0 = 1.408;
            p.gammaInf = 0.827;
            p.beta = 0.826; 
            p.a0 = 0.0002121; # [K^-1]
            p.m = 1.891;
            p.g = 1.339;
            p.G0 = 0;
            p.Gp = 0;
            p.eta = 0;
            p.EOS = 1; # Holzapfel EOS
            p.en = p.EOS
        elif name=='FeSi':
            p.name = name # from Sata et al. 2010
            p.T0 = 300 # [K]
            p.n = 2
            p.V0 = 10.685*mol*1e-24*p.n # Angstrom^3/atom -> cm^3/mol; FeSi -> Fe0.5Si0.5, hence V0/n
            p.M = Fe+Si # [g mol^-1]
            p.KT0 = 221.7 # [GPa]
            p.KTp = 4.167 # [1]
            p.theta0 = 44.574 # [K] # values from Fe hcp phase, not used in the end
            p.gamma0 = 1.408 # negative values for theta0 and gamma0 for BM3 with Holzapfel energy
            p.gammaInf = 0.827
            p.beta = 0.826
            p.en = 1 # use Holzapfel energy in BM3 EOS
            p.G0 = 0;
            p.Gp = 0;
            p.eta = 0
        elif name=='Fe3C':
            p.name = name # from Sata et al. 2010
            p.T0 = 300 # [K]
            p.n = 4
            p.V0 = 9.341*mol*1e-24*p.n # Angstrom^3/atom -> cm^3/mol
            p.M = 3*Fe+C # [g mol^-1]
            p.KT0 = 290.0 # [GPa]
            p.KTp = 3.76 # [1]
            p.theta0 = 44.574 # [K] # values from Fe hcp phase
            p.gamma0 = 1.408 # negative values for theta0 and gamma0 for BM3 with Holzapfel energy
            p.gammaInf = 0.827
            p.beta = 0.826
            p.en = 1 # use Holzapfel energy in BM3 EOS
            p.G0 = 0;
            p.Gp = 0;
            p.eta = 0
        elif name=='FeO': # actually Fe0.95 O
            p.name = name # from Sata et al. 2010
            p.T0 = 300 # [K]
            p.n = 1.95
            p.V0 = 10.113*mol*1e-24*p.n # Angstrom^3/atom -> cm^3/mol# comparison with wu (FeO) shows *p.n is needed
            p.M = 0.95*Fe+O # [g mol^-1]
            p.KT0 = 154.0 # [GPa]
            p.KTp = 4.04 # [1]
            p.theta0 = 44.574 # [K] # values from Fe hcp phase
            p.gamma0 = 1.408 # negative values for theta0 and gamma0 for BM3 with Holzapfel energy
            p.gammaInf = 0.827
            p.beta = 0.826
            p.en = 1 # use Holzapfel energy in BM3 EOS
            p.G0 = 0;
            p.Gp = 0;
            p.eta = 0
        elif name=='FeS': # FeS VII phase, valid from 180 GPa on
            p.name = name # from Sata et al. 2010
            p.T0 = 300 # [K]
            p.n = 2
            p.V0 = 11.931*mol*1e-24*p.n # Angstrom^3/atom -> cm^3/mol
            p.M = Fe+S # [g mol^-1]
            p.KT0 = 148.0 # [GPa]
            p.KTp = 4.53 # [1]
            p.theta0 = 44.574 # [K] # values from Fe hcp phase
            p.gamma0 = 1.408 # negative values for theta0 and gamma0 for BM3 with Holzapfel energy
            p.gammaInf = 0.827
            p.beta = 0.826
            p.en = 1 # use Holzapfel energy in BM3 EOS
            p.G0 = 0
            p.Gp = 0
            p.eta = 0
        elif name=='h2o':
            p.V0 = 18 # M/rho[g/cm^3]
            p.M = 2*H+O
        elif name=='iceVI':
            p.M = 2*H+O
            p.n = 3
            p.V0 = p.M/1.2717 # M/rho[g/cm^3]
            p.KT0 = 14.05  # [GPa]
            p.KTp = 4.0  # [1]
            p.EOS = 2
            p.theta0 = 1
            p.gamma0 = 0
            p.gammaInf = 0
            p.q0 = 0
            p.G0 = 0
            p.Gp = 0
            p.beta = 0
            p.en = 0
            p.G0 = 0
            p.Gp = 0
            p.eta = 0
        elif name == 'iceVII':
            p.M = 2 * H + O
            p.n = 3
            p.V0 = p.M / 1.4424  # M/rho[g/cm^3]
            p.KT0 = 20.15  # [GPa]
            p.KTp = 4.0  # [1]
            p.EOS = 2
            p.theta0 = 1
            p.gamma0 = 0
            p.gammaInf = 0
            p.q0 = 0
            p.G0 = 0
            p.Gp = 0
            p.beta = 0
            p.en = 0
            p.G0 = 0
            p.Gp = 0
            p.eta = 0
        elif name == 'iceX':
            p.M = 2 * H + O
            p.n = 3
            p.V0 = p.M / 2.3186  # M/rho[g/cm^3]
            p.KT0 = 162.8  # [GPa]
            p.KTp = 4.42  # [1]
            p.EOS = 2
            p.theta0 = 1
            p.gamma0 = 0
            p.gammaInf = 0
            p.q0 = 0
            p.G0 = 0
            p.Gp = 0
            p.beta = 0
            p.en = 0
            p.G0 = 0
            p.Gp = 0
            p.eta = 0

        #                         rho,p,Tad,rho0,K0,Kp0,Kpinf,gr0,gr_inf,theta0,beta,Mmol,n,EOS,Vers,Iron
        # ice VI: call DensityEOS(rho,p_GPa,T,1271.7,14.05,4.0,0.0,0.0,0.0,0.0,0.0,0.0,0,1,0,0) # Bezacier et al., 2014
        # ice VII: call DensityEOS(rho,p_GPa,T,1442.4,20.15,4.0,0.0,0.0,0.0,0.0,0.0,0.0,0,1,0,0) # Bezacier et al., 2014
        # ice X: call DensityEOS(rho,p_GPa,T,2318.6,162.8,4.42,0.0,0.0,0.0,0.0,0.0,0.0,0,1,0,0) # Journaux et al., 2014

        else:
            print('ERROR, species ',name, ' not known!')

        p.V0 = p.V0 * 1000; # cm^3/mol in mm^3/mol    
        
        return 
