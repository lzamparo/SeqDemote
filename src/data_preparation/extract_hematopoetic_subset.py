import os
import pandas
import numpy as np

### Grab all cells within the hematopoetic lineage out of the Roadmap data used for Alvaro's paper

os.chdir(os.path.expanduser('~/projects/SeqDemote/data/DNase'))


# read both peaks data file, DNase counts file
peaks = pandas.read_csv("peaksTable.txt", sep="\t")
counts = pandas.read_csv("DNaseCnts.txt", sep="\t")

# establish a dictionary to return cell type codes in the form of an np.array
activity_dict = {}
activity_dict['H1hesc'] = '0'
activity_dict['CD34'] = '1'
activity_dict['CD14'] = '2'
activity_dict['CD56'] = '3'
activity_dict['CD3'] = '4'
activity_dict['CD19'] = '5'
    
    
#chr13	19144320	19144920	.	1	+	4
#chr13	19168795	19169395	.	1	+	72
#chr13	19170000	19170600	.	1	+	25
#chr13	19172116	19172716	.	1	+	2,8,10,21,26,29,48,118    
        
# Convert Alvaro's data set into a bedfile format expected by preprocess_features.py
def extend_peak(start,end, length=600):
    ''' If start,end is less than length, extend them.  If start,end is more than length, cut them. '''
    peak_length = end - start
    discrepancy = length - peak_length
    adjustment = np.abs(discrepancy) // 2
    offset = np.abs(discrepancy) % 2
    new_start = start - np.sign(discrepancy) * adjustment
    new_end = end +  np.sign(discrepancy) *(adjustment + offset)
    assert(new_end - new_start == length)
    return new_start, new_end
    

def alvaro_to_bed_format(line):
    '''Convert a line from Alvaros peaksTable.txt to BED format '''
    accessPattern = line['accessPattern']
    active_in = ",".join([activity_dict[e] for e in accessPattern.split('-')])
    new_start, new_end = extend_peak(line['start'], line['end'])
    outline = '\t'.join([line['chr'], str(new_start), str(new_end), str(line['peakID']), '1', '+', active_in])
    return outline
    
outfile = open('hematopoetic_peaks.bed','w')    
for index, line in peaks.iterrows():
    print(alvaro_to_bed_format(line), file=outfile)

outfile.close()    


# Use bedtools to extract fasta formatted sequences based on the bedtools format


# Make an activity table by parsing Alvaro's file again

    #8988T	AoSMC	Chorion	CLL	Fibrobl	FibroP	Gliobla	GM12891	GM12892	GM18507	GM19238	GM19239	GM19240	H9ES	HeLa-S3_IFNa4h	Hepatocytes	HPDE6-E6E7	HSMM_emb	HTR8svn	Huh-7.5	Huh-7	iPS	Ishikawa_Estradiol	Ishikawa_4OHTAM	LNCaP_androgen	MCF-7_Hypoxia	Medullo	Melano	Myometr	Osteobl	PanIsletD	PanIslets	pHTE	ProgFib	RWPE1	Stellate	T-47D	CD4_Th0	Urothelia	Urothelia_UT189	AG04449	AG04450	AG09309	AG09319	AG10803	AoAF	BE2_C	BJ	Caco-2	CD20+	CD34+	CMK	GM06990	GM12864	GM12865	H7-hESC	HAc	HAEpiC	HA-h	HA-sp	HBMEC	HCF	HCFaa	HCM	HConF	HCPEpiCHCT-116	HEEpiC	HFF	HFF-Myc	HGF	HIPEpiC	HL-60	HMF	HMVEC-dAd	HMVEC-dBl-Ad	HMVEC-dBl-Neo	HMVEC-dLy-Ad	HMVEC-dLy-Neo	HMVEC-dNeo	HMVEC-LBl	HMVEC-LLy	HNPCEpiC	HPAEC	HPAF	HPdLF	HPF	HRCEpiC	HRE	HRGEC	HRPEpiC	HVMF	Jurkat	Monocytes-CD14+	NB4	NH-A	NHDF-Ad	NHDF-neo	NHLF	NT2-D1	PANC-1	PrEC	RPTEC	SAEC	SKMC	SK-N-MC	SK-N-SH_RA	Th2	WERI-Rb-1	WI-38	WI-38_4OHTAM	A549	GM12878	H1-hESC	HeLa-S3	HepG2	HMEC	HSMM	HSMMtube	HUVEC	K562	LNCaP	MCF-7	NHEK	Th1	LNG.IMR90 	ESC.H9 	ESC.H1 	IPSC.DF.6.9 	IPSC.DF.19.11 	ESDR.H1.NEUR.PROG 	ESDR.H1.BMP4.MESO 	ESDR.H1.BMP4.TROP 	ESDR.H1.MSC 	BLD.CD3.PPC 	BLD.CD3.CPC 	BLD.CD14.PC 	BLD.MOB.CD34.PC.M 	BLD.MOB.CD34.PC.F 	BLD.CD19.PPC 	BLD.CD56.PC 	SKIN.PEN.FRSK.FIB.01 	SKIN.PEN.FRSK.FIB.02 	SKIN.PEN.FRSK.MEL.01 	SKIN.PEN.FRSK.KER.02 	BRST.HMEC.35 	THYM.FET 	BRN.FET.F 	BRN.FET.M 	MUS.PSOAS 	MUS.TRNK.FET 	MUS.LEG.FET 	HRT.FET 	GI.STMC.FET 	GI.S.INT.FET 	GI.L.INT.FET 	GI.S.INT 	GI.STMC.GAST 	KID.FET 	LNG.FET 	OVRY 	ADRL.GLND.FET 	PLCNT.FET 	PANC
#chr13:19144320-19144920(+)	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
#chr13:19168795-19169395(+)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
#chr13:19170000-19170600(+)	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	00	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0

# establish a dictionary to return indicators in the form of an np.array
pattern_dict = {}
pattern_dict['H1hesc'] = np.asarray([1,0,0,0,0,0], dtype=int)
pattern_dict['CD34'] = np.asarray([0,1,0,0,0,0], dtype=int)
pattern_dict['CD14'] = np.asarray([0,0,1,0,0,0], dtype=int)
pattern_dict['CD56'] = np.asarray([0,0,0,1,0,0], dtype=int)
pattern_dict['CD3'] = np.asarray([0,0,0,0,1,0], dtype=int)
pattern_dict['CD19'] = np.asarray([0,0,0,0,0,1], dtype=int)

def parse_access_pattern(pattern):
    ''' The access pattern in Alvaro's data is a dash delimited string indicating in which cell-types this particular peak is accessible.
    I need to return this as an int8 nparray
    e.g if pattern is H1hesc-CD34-CD14-CD56-CD3-CD19, output is np.asarray([1,1,1,1,1,1])
        if pattern is CD34-CD14, output is np.asarray([0,1,1,0,0,0])
    
    '''
    arrays = tuple([pattern_dict[d] for d in pattern.split('-')])
    return np.sum(np.vstack(arrays), axis=0)


# Encode the fasta sequences, activity table in a tensor, store in an h5 file



