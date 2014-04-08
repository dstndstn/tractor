#include <stdio.h>
#include <math.h>
char c1[14],c2[14],head[72][80];
double a8[8],b8[8],pi,cx,cy;
FILE *fp,*fp1;
int  n1,n2;

main(ac,av)
int ac; char *av[];
{
  char a[50],f1[50];
  double x,y,xa,xd,epoch,alpha,delta;
  int i,k;
  pi=4.*atan(1.0); cx=pi/12.; cy=pi/180.;
  if(ac<2){
    printf("\n\t*****  z_9_ccd  R.A.,Dec. --> X,Y *****\n");
    printf("\n\t Usage: z_9ad2xy fitsfile a1 d1 [a2 d2 ...]");
    printf("\n\t        z_9ad2xy batch_file\n");  
    printf("\n\t in batch file:");
    printf("\n\t   fitsfile_name");
    printf("\n\t   epoch");
    printf("\n\t   a1 d1");
    printf("\n\t   ...\n");
    printf("\n\t                                   2011,9,6\n");
    exit(0);
  }
  epoch=2000.0;
  strcpy(f1,av[1]);
  if(ac==2){
    fp1=fopen(f1,"r");
    if(fp1==0){ printf("\ntbatch_file not found!\n"), exit(0); }
    fgets(f1,50,fp1); f1[strlen(f1)-1]=0;
    fgets(a,50,fp1);  sscanf(a,"%lf",&epoch);
  }
  fp=fopen(f1,"rb"); if(fp==0){printf("\nfits not found %s\n",f1); exit(0); }
  fread(head,72,80,fp); fclose(fp);
  k=indexpos(head,"NAXIS1  ",72);
  sscanf(&head[k++][23],"%d",&n1);
  sscanf(&head[k++][23],"%d",&n2);
//  if(n1!=4096 || n2!=4032){ printf("\nnot BOK_ccd !\n\n"); exit(0); }
  k=indexpos(head,"A81     ",72);
  if(k==72){ printf("fits not coordination !\n"); exit(0); }
  for(i=0;i<8;i++)sscanf(&head[k+i][10],"%le",&a8[i]);
  xytoad(a8,b8);

  if(ac==2){
l30:
    fgets(a,50,fp1); if(feof(fp1))exit(0);
    if(a[0]=='#')goto l30; 
    if(strlen(a)<3)goto l30;
    sscanf(a,"%s %s",f1,a);
    chs(f1,&xa); chs(a,&xd);
    astprs2(xa,xd,epoch,&alpha,&delta,2000.);
    rad_xy(a8[6],a8[7],alpha,delta,&x,&y,1,0);
    toms2(alpha,c1,1);  toms2(delta,c2,0);
    printf("%s %s (%6.1lf)--->%9.2lf %9.2lf\n",c1,c2,epoch,x,y);
    goto l30;
  }
  k=2;
l10:
  strcpy(a,av[k++]); chs(a,&alpha);
  strcpy(a,av[k++]); chs(a,&delta);
  rad_xy(a8[6],a8[7],alpha,delta,&x,&y,1,0);
  toms2(alpha,c1,1);  toms2(delta,c2,0);
  printf("%s %s (%6.1lf)--->%9.2lf %9.2lf\n",c1,c2,epoch,x,y);
  if(ac>k+1)goto l10;
}

chs(a,y)
char *a; double *y;
{
  double x;
  char b[20];
  int i,j,k;
  strcpy(b,a); k=strlen(b);
  j=1; for(i=0;i<k;i++){ if(b[i]==32)continue; if(b[i]=='-')j=-1; break; }
  for(i=0;i<k;i++)if(b[i]<'.' || b[i]> '9')b[i]=32;
  i=k=0; x=0.; sscanf(b,"%d %d %lf",&i,&k,&x);
  x=k/60.+x/3600.+i;
  *y=x*j;
}

astprs2(ra1, dc1, ep1, ra2, dc2, ep2)   // RA in hour, DEC in degree
double  ra1,dc1,ep1,*ra2,*dc2,ep2;      // only ra2,dc2   is output
{
  double r0[3],r1[3],p[3][3],arc;
  double r2,d2;

  arc=45./atan(1.);
  *ra2=ra1; *dc2=dc1;
  if(ep1 == ep2)return;
  r2 =ra1*15./arc; d2 =dc1/arc;
  r0[0]=cos(r2)*cos(d2); r0[1]=sin(r2)*cos(d2); r0[2]=sin(d2);
  if(ep1 != 2000.){
    astrox(ep1, p);
    r1[0] = p[0][0] * r0[0] + p[0][1] * r0[1] + p[0][2] * r0[2];
    r1[1] = p[1][0] * r0[0] + p[1][1] * r0[1] + p[1][2] * r0[2];
    r1[2] = p[2][0] * r0[0] + p[2][1] * r0[1] + p[2][2] * r0[2];
    r0[0] = r1[0]; r0[1] = r1[1]; r0[2] = r1[2];
  }
  if(ep2 != 2000.){
    astrox(ep2, p);
    r1[0] = p[0][0] * r0[0] + p[1][0] * r0[1] + p[2][0] * r0[2];
    r1[1] = p[0][1] * r0[0] + p[1][1] * r0[1] + p[2][1] * r0[2];
    r1[2] = p[0][2] * r0[0] + p[1][2] * r0[1] + p[2][2] * r0[2];
    r0[0] = r1[0];    r0[1] = r1[1];    r0[2] = r1[2];
  }
  *ra2  = atan2(r0[1], r0[0])/15.*arc;
  *dc2 = asin(r0[2])*arc;
  if(*ra2<0)*ra2+=24.;
}

astrox (epoch, p)
double epoch,p[3][3];
{
  double t,a,b,c,ca,cb,cc,sa,sb,sc,arc;
  arc=45./atan(1.);
  astjuy(epoch,&t);
  t = (t - 2451545.0) / 36525.;
  a = t * (0.6406161 + t * (0.0000839 + t * 0.0000050));
  b = t * (0.6406161 + t * (0.0003041 + t * 0.0000051));
  c = t * (0.5567530 - t * (0.0001185 + t * 0.0000116));
  ca = cos (a/arc);
  sa = sin (a/arc);
  cb = cos (b/arc);
  sb = sin (b/arc);
  cc = cos (c/arc);
  sc = sin (c/arc);
  p[0][0] = ca * cb * cc - sa * sb;
  p[1][0] = -sa * cb * cc - ca * sb;
  p[2][0] = -cb * sc;
  p[0][1] = ca * sb * cc + sa * cb;
  p[1][1] = -sa * sb * cc + ca * cb;
  p[2][1] = -sb * sc;
  p[0][2] = ca * sc;
  p[1][2] = -sa * sc;
  p[2][2] = cc;
}

astjuy (epoch,t)
double epoch,*t;
{
  double jd;
  int year,centuy;
  year = epoch - 1;
  centuy = year / 100;
  jd =1721425.5+365.*year-centuy+ year/4 + centuy/4;
  year=epoch;
  *t= jd + (epoch - year) * 365.25;
}

toms2(aa,c,k)
double aa;           // input
char c[14];          // output
int k;               // input
/*
 hour or degree to char_line
 if k=1 (hour case) in **:**:**.***
    k=0 (degree       -**:**:**.**
*/
{
  int  i,j;
  double a,b,x;
  a=aa;
  if(k==1){ if(a>24.)a-=24.; if(a<0.)a+=24.; }
  c[0]=32; if(a<0.){ a=-a; c[0]='-'; }
  i=a; b=(a-i)*60.;
  j=b; x=(b-j)*60.;
  if(j==60){ j=0; i++; }
  if(k==1)sprintf(&c[1],"%2.2d:%2.2d:%6.3f",i,j,x);
  if(k==0)sprintf(&c[1],"%2.2d:%2.2d:%5.2f",i,j,x);
  if(c[7]==32)c[7]='0';   if(c[8]==32)c[8]='0';
  if(c[7]=='6'){
    c[7]='0'; j++; sprintf(&c[4],"%2.2d",j);
    if(c[4]=='6'){
      c[4]='0'; i++; sprintf(&c[1],"%2.2d",i);
    }
    c[3]=':'; c[6]=':';
  }
}

xytoad(x,a)
double *x,*a;
{
  double z;
  z=x[0]*x[3]-x[2]*x[1];
  a[0]= x[3]/z;
  a[1]=-x[1]/z;
  a[2]=-x[2]/z;
  a[3]= x[0]/z;
  a[4]= (x[2]*x[5]-x[4]*x[3])/z;
  a[5]= (x[4]*x[1]-x[0]*x[5])/z;
}

#define bok  48.
rad_xy(ac,dc,ra,de,x,y,type,k)
double ac,dc,ra,de,*x,*y;  int type,k;
{
  double xi,xn, tmp,rar,der;
  rar=ra*cx; der=de*cy;
/*                         // same discribe formula as following 3 lines
  double sd,cd,td,co;      // china bai_ke_quan_shu (astronmy) p.552
  sd=sin(dc);  cd=cos(dc);  td=tan(der);
  co=cos(rar-ac);  tmp=sd*td+cd*co;
  xi=sin(rar-ac)/tmp;
  xn=(cd*td-sd*co)/tmp;
*/
  tmp=atan(tan(der)/cos(rar-ac));
  xi=cos(tmp)*tan(rar-ac)/cos(tmp-dc);
  xn=tan(tmp-dc);
  if(type==1){     // schmidt TELESCOPE
    tmp=sqrt(xi*xi+xn*xn);
    if(tmp!=0.){
      tmp=atan(tmp)/tmp;
      xi*=tmp; xn*=tmp;
    }
  }
  if(type==2){    //  BOK telescope
    tmp=1.+bok*(xi*xi+xn*xn);
    xi*=tmp; xn*=tmp;
  }
  if(k==0){
    *x=b8[0]*xi+b8[2]*xn+b8[4];
    *y=b8[1]*xi+b8[3]*xn+b8[5];
  }
  if(k==1){ *x=xi; *y=xn;}
}

xy_rad(ac,dc,x,y,ra,de,type,k)
double ac,dc,x,y,*ra,*de;  int type,k;
{
  double xi,xn,tmp,tmp1;
  double x1,y1,r1,d1;
  int i;
  xi=a8[0]*x+a8[2]*y+a8[4];
  xn=a8[1]*x+a8[3]*y+a8[5];
  if(type==1){
    tmp=sqrt(xi*xi+xn*xn);
    if(tmp!=0.){
      tmp=tan(tmp)/tmp;
      xi*=tmp;  xn*=tmp;
    }
  }
  if(type==2){
    tmp=1.+bok*(xi*xi+xn*xn);
    for(i=0;i<4;i++){
      x1=xi/tmp; y1=xn/tmp;
      tmp=tan(dc);  tmp1=1.-y1*tmp;
      r1=atan(x1/cos(dc)/tmp1);
      d1=atan((y1+tmp)*cos(r1)/tmp1);
      r1+=ac;
      tmp=atan(tan(d1)/cos(r1-ac));
      x1=cos(tmp)*tan(r1-ac)/cos(tmp-dc);
      y1=tan(tmp-dc);
      tmp=1.+bok*(x1*x1+y1*y1);
//                                 printf("%lf\n",tmp);
    }
    xi/=tmp; xn/=tmp;
  }
  tmp=tan(dc);  tmp1=1.-xn*tmp;
  *ra=atan(xi/cos(dc)/tmp1);
  *de=atan((xn+tmp)*cos(*ra)/tmp1);
  *ra+=ac;
  tmp=(*de)-dc; if(tmp<0.)tmp=-tmp;
  if(tmp>2.){  *de=-(*de);  *ra+=pi; }
  if(*ra<0.)*ra+=(pi+pi);
  if(k==0){ *ra/=cx; *de/=cy; }
}

indexpos(head,f1,n)
char head[][80],f1[8];
int n;
{
  int i,j;
  for(i=0;i<n;i++){
    for(j=0;j<8;j++) if(head[i][j]!=f1[j])goto l10;
    return i;
l10:
    continue;
  }
  return i;
}


