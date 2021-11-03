#include <stdio.h>
#include <string.h>




#define VAL_1X     -128
#define VAL_2X     VAL_1X,  VAL_1X
#define VAL_3X     VAL_1X,  VAL_1X, VAL_1X
#define VAL_4X     VAL_2X,  VAL_2X
#define VAL_8X     VAL_4X,  VAL_4X
#define VAL_16X    VAL_8X,  VAL_8X
#define VAL_32X    VAL_16X, VAL_16X
#define VAL_64X    VAL_32X, VAL_32X
#define VAL_128X   VAL_64X, VAL_64X
static const char nuc_table[256] = { VAL_64X, VAL_32X, VAL_1X, 0, VAL_1X, 1, VAL_3X, 2, VAL_8X, VAL_4X, 3, VAL_128X, VAL_8X, VAL_3X };

unsigned char compl[256] = "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnntvghnncdnnmnknnnnynanbnnrnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn";

unsigned char aa_table[65] = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV#Y+YSSSS*CWCLFLFX";


int main(int argc, char** argv){
    unsigned int idx;
	char dna[3];
	printf("Value: %c (%i) -> %c\n", 'r', 'r', compl['r']);
	printf("Value: %c (%i) -> %c\n", 'y', 'y', compl['y']);
	printf("Value: %c (%i) -> %c\n", 'k', 'k', compl['k']);
	printf("Value: %c (%i) -> %c\n", 'm', 'm', compl['m']);
	printf("Value: %c (%i) -> %c\n", 'b', 'b', compl['b']);
	printf("Value: %c (%i) -> %c\n", 'v', 'v', compl['v']);
	printf("Value: %c (%i) -> %c\n", 'd', 'd', compl['d']);
	printf("Value: %c (%i) -> %c\n", 'h', 'h', compl['h']);
	return 0;

	strcpy(dna,argv[1]);
    if(1){
        idx = nuc_table[dna[0]];
        idx = idx*4 + nuc_table[dna[1]];
        idx = idx*4 + nuc_table[dna[2]];
    }else{
        idx = nuc_table[compl[dna[2]]];
        idx = idx*4 + nuc_table[compl[dna[1]]];
        idx = idx*4 + nuc_table[compl[dna[0]]];
    }
    if(idx <= 63){
        printf("Value: %i %c\n", idx, aa_table[idx]);
    }else{
		//long val = (dna[0] << 32L) | (dna[1] << 16) | dna[2];
		int val = (dna[0] + (dna[1] * 255) + dna[2] * 255 * 255);
		switch(val){
			case 7437682 : idx = 0; break;
			case 7892857 : idx = 1; break;
			case 7438192 : idx = 4; break;
			case 7893367 : idx = 4; break;
			case 6983017 : idx = 4; break;
			case 7113067 : idx = 4; break;
			case 6397792 : idx = 4; break;
			case 7698292 : idx = 4; break;
			case 6527842 : idx = 4; break;
			case 6787942 : idx = 4; break;
			case 7439212 : idx = 8; break;
			case 7894387 : idx = 9; break;
			case 7897702 : idx = 12; break;
			case 7117402 : idx = 12; break;
			case 6792277 : idx = 12; break;
			case 7437684 : idx = 16; break;
			case 7892859 : idx = 17; break;
			case 7438194 : idx = 20; break;
			case 7893369 : idx = 20; break;
			case 6983019 : idx = 20; break;
			case 7113069 : idx = 20; break;
			case 6397794 : idx = 20; break;
			case 7698294 : idx = 20; break;
			case 6527844 : idx = 20; break;
			case 6787944 : idx = 20; break;
			case 7439214 : idx = 24; break;
			case 7894389 : idx = 24; break;
			case 6984039 : idx = 24; break;
			case 7114089 : idx = 24; break;
			case 6398814 : idx = 24; break;
			case 7699314 : idx = 24; break;
			case 6528864 : idx = 24; break;
			case 6788964 : idx = 24; break;
			case 7442529 : idx = 28; break;
			case 7897704 : idx = 28; break;
			case 6987354 : idx = 28; break;
			case 7117404 : idx = 28; break;
			case 6402129 : idx = 28; break;
			case 7702629 : idx = 28; break;
			case 6532179 : idx = 28; break;
			case 6792279 : idx = 28; break;
			case 7437688 : idx = 32; break;
			case 7892863 : idx = 33; break;
			case 7438198 : idx = 36; break;
			case 7893373 : idx = 36; break;
			case 6983023 : idx = 36; break;
			case 7113073 : idx = 36; break;
			case 6397798 : idx = 36; break;
			case 7698298 : idx = 36; break;
			case 6527848 : idx = 36; break;
			case 6787948 : idx = 36; break;
			case 7439218 : idx = 40; break;
			case 7894393 : idx = 40; break;
			case 6984043 : idx = 40; break;
			case 7114093 : idx = 40; break;
			case 6398818 : idx = 40; break;
			case 7699318 : idx = 40; break;
			case 6528868 : idx = 40; break;
			case 6788968 : idx = 40; break;
			case 7442533 : idx = 44; break;
			case 7897708 : idx = 44; break;
			case 6987358 : idx = 44; break;
			case 7117408 : idx = 44; break;
			case 6402133 : idx = 44; break;
			case 7702633 : idx = 44; break;
			case 6532183 : idx = 44; break;
			case 6792283 : idx = 44; break;
			case 7437701 : idx = 56; break;
			case 7892876 : idx = 49; break;
			case 7438211 : idx = 9; break;
			case 7893386 : idx = 9; break;
			case 6983036 : idx = 9; break;
			case 7113086 : idx = 9; break;
			case 6397811 : idx = 9; break;
			case 7698311 : idx = 9; break;
			case 6527861 : idx = 9; break;
			case 6787961 : idx = 9; break;
			case 7894406 : idx = 57; break;
			case 7442546 : idx = 28; break;
			case 7897721 : idx = 61; break;
			case 6336611 : idx = 56; break;
			case 6337126 : idx = 28; break;
			case 6727276 : idx = 28; break;
			case 7442551 : idx = 28; break;
			case 6333799 : idx = 8; break;
			case 6723949 : idx = 8; break;
			case 7439224 : idx = 8; break;
			default      : idx = 64;
		}
        printf("Value: %i %c\n", idx, aa_table[idx]);
    }
}
