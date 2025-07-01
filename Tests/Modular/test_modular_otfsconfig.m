clear
clc

oc = OTFSConfig();

%{
set the frame type
%}
oc.setFrame(OTFSConfig.FRAME_TYPE_GIVEN,  3, 5, "zp_len", 6);
oc.setFrame(OTFSConfig.FRAME_TYPE_FULL,   3, 5);
oc.setFrame(OTFSConfig.FRAME_TYPE_CP,     3, 5);
oc.setFrame(OTFSConfig.FRAME_TYPE_ZP,     3, 5, "zp_len", 4);

%{
set the pulse type
%}
oc.setPul(OTFSConfig.PUL_TYPE_IDEAL);
oc.setPul(OTFSConfig.PUL_TYPE_RECTA);

%{
set the pilot
%}
oc.setPil(OTFSConfig.PIL_TYPE_GIVEN);
oc.setPil(OTFSConfig.PIL_TYPE_NO);
oc.setPil(OTFSConfig.PIL_TYPE_EM_MID, "pk_len", 2, "pl_len", 2);
oc.setPil(OTFSConfig.PIL_TYPE_SP_MID, "pk_len", 2, "pl_len", 2);
oc.setPil(OTFSConfig.PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY, "pk_len", 2, "pl_len", 2, "pl_num", 2);