import numpy as np
import sqlite3 as sql
import pymonetdb

from sql_sparse_einsum import sql_einsum_query_opt
from tableauhyperapi import HyperProcess, Telemetry, CreateMode, Connection
from timeit import default_timer as timer


# example problem
def create_sat_952():
    dt = np.float64
    E12 = np.array([[[1., 1.], [1., 1.]], [[1., 1.], [0., 1.]]], dt)
    E11 = np.array([[[1., 1.], [1., 1.]], [[1., 0.], [1., 1.]]], dt)
    E10 = np.array([[[1., 1.], [1., 1.]], [[0., 1.], [1., 1.]]], dt)
    E0 = np.array([0., 1.], dt)
    E4 = np.array([[1., 1.], [0., 1.]], dt)
    E5 = np.array([[1., 1.], [1., 0.]], dt)
    arrays = [E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E5, E4, E5, E4, E5, E4, E5, E4, E5, E4, E5, E4, E12, E11, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E0, E12, E11, E0, E12, E11, E0, E5, E4, E5, E4, E12, E11, E10, E12, E10, E10, E12, E10, E0, E5, E4, E12, E11, E10, E12, E10, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E5, E4, E12, E11, E10, E12, E10, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E5, E4, E5, E4, E5, E4, E5, E4, E5, E4, E12, E11, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E0, E5, E4, E5, E4, E5, E4, E12, E11, E10, E12, E10, E10, E12, E10, E10, E12, E10, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E5, E4, E5, E4, E5, E4, E5, E4, E5, E4, E12, E11, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E0, E5, E4, E5, E4, E5, E4, E5, E4, E5, E4, E12, E11, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E10, E12, E10, E0, E5, E4, E12, E11, E10, E12, E10, E0, E5, E4, E12, E11, E10, E12, E10, E0, E12, E11, E0, E12, E11, E0, E12, E11, E0, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4, E5, E4, E4, E5, E4, E4, E4, E4, E4, E4, E4, E4, E4, E4]
    format_string = 'dcb,dcb,d,gfe,gfe,g,jih,jih,j,mlk,mlk,m,pon,pon,p,srq,srq,s,vut,vut,v,yxw,yxw,y,Hz,HG,IA,IH,JB,JI,KC,KJ,LD,LK,ME,ML,NGz,NGz,OAN,OAH,ONH,PBO,PBI,POI,QCP,QCJ,QPJ,RDQ,RDK,RQK,SER,SEL,SRL,TFS,TFM,TSM,T,WVU,WVU,W,ZYX,ZYX,Z,ÄÀ,ÄÃ,ÅÁ,ÅÄ,ÆÃÀ,ÆÃÀ,ÇÁÆ,ÇÁÄ,ÇÆÄ,ÈÂÇ,ÈÂÅ,ÈÇÅ,È,ÌÉ,ÌË,ÍËÉ,ÍËÉ,ÎÊÍ,ÎÊÌ,ÎÍÌ,Î,ÑÐÏ,ÑÐÏ,Ñ,ÔÓÒ,ÔÓÒ,Ô,×ÖÕ,×ÖÕ,×,ÚÙØ,ÚÙØ,Ú,ÝÜÛ,ÝÜÛ,Ý,àßÞ,àßÞ,à,äá,äã,åãá,åãá,æâå,æâä,æåä,æ,éèç,éèç,é,ìëê,ìëê,ì,ïîí,ïîí,ï,òñð,òñð,ò,úó,úù,ûô,ûú,üõ,üû,ýö,ýü,þ÷,þý,ÿùó,ÿùó,Āôÿ,Āôú,Āÿú,āõĀ,āõû,āĀû,Ăöā,Ăöü,Ăāü,ă÷Ă,ă÷ý,ăĂý,Ąøă,Ąøþ,Ąăþ,Ą,Ċą,Ċĉ,ċĆ,ċĊ,Čć,Čċ,čĉą,čĉą,ĎĆč,ĎĆĊ,ĎčĊ,ďćĎ,ďćċ,ďĎċ,ĐĈď,ĐĈČ,ĐďČ,Đ,ēĒđ,ēĒđ,ē,ĖĕĔ,ĖĕĔ,Ė,ęĘė,ęĘė,ę,ĜěĚ,ĜěĚ,Ĝ,ğĞĝ,ğĞĝ,ğ,ĢġĠ,ĢġĠ,Ģ,ĥĤģ,ĥĤģ,ĥ,ĨħĦ,ĨħĦ,Ĩ,īĪĩ,īĪĩ,ī,ĮĭĬ,ĮĭĬ,Į,ıİį,ıİį,ı,ĴĳĲ,ĴĳĲ,Ĵ,ķĶĵ,ķĶĵ,ķ,ĺĹĸ,ĺĹĸ,ĺ,ĽļĻ,ĽļĻ,Ľ,ŀĿľ,ŀĿľ,ŀ,ŃłŁ,ŃłŁ,Ń,ņŅń,ņŅń,ņ,ŉňŇ,ŉňŇ,ŉ,ŌŋŊ,ŌŋŊ,Ō,ŏŎō,ŏŎō,ŏ,ŒőŐ,ŒőŐ,Œ,ŕŔœ,ŕŔœ,ŕ,ŘŗŖ,ŘŗŖ,Ř,śŚř,śŚř,ś,ŞŝŜ,ŞŝŜ,Ş,šŠş,šŠş,š,ŤţŢ,ŤţŢ,Ť,ŧŦť,ŧŦť,ŧ,ŪũŨ,ŪũŨ,Ū,ŭŬū,ŭŬū,ŭ,ŰůŮ,ŰůŮ,Ű,ųŲű,ųŲű,ų,ŶŵŴ,ŶŵŴ,Ŷ,ŹŸŷ,ŹŸŷ,Ź,żŻź,żŻź,ż,ſžŽ,ſžŽ,ſ,ƂƁƀ,ƂƁƀ,Ƃ,ƅƄƃ,ƅƄƃ,ƅ,ƈƇƆ,ƈƇƆ,ƈ,ƋƊƉ,ƋƊƉ,Ƌ,Ǝƍƌ,Ǝƍƌ,Ǝ,ƑƐƏ,ƑƐƏ,Ƒ,ƔƓƒ,ƔƓƒ,Ɣ,ƗƖƕ,ƗƖƕ,Ɨ,ƚƙƘ,ƚƙƘ,ƚ,ƝƜƛ,ƝƜƛ,Ɲ,ƠƟƞ,ƠƟƞ,Ơ,ƣƢơ,ƣƢơ,ƣ,ƦƥƤ,ƦƥƤ,Ʀ,ƩƨƧ,ƩƨƧ,Ʃ,Ƭƫƪ,Ƭƫƪ,Ƭ,ƯƮƭ,ƯƮƭ,Ư,ƲƱư,ƲƱư,Ʋ,ƵƴƳ,ƵƴƳ,Ƶ,ƸƷƶ,ƸƷƶ,Ƹ,ƻƺƹ,ƻƺƹ,ƻ,ƾƽƼ,ƾƽƼ,ƾ,ǁǀƿ,ǁǀƿ,ǁ,Ǆǃǂ,Ǆǃǂ,Ǆ,Ǉǆǅ,Ǉǆǅ,Ǉ,Ǌǉǈ,Ǌǉǈ,Ǌ,Ǎǌǋ,Ǎǌǋ,Ǎ,ǐǏǎ,ǐǏǎ,ǐ,ǓǒǑ,ǓǒǑ,Ǔ,ǖǕǔ,ǖǕǔ,ǖ,ǙǘǗ,ǙǘǗ,Ǚ,ǜǛǚ,ǜǛǚ,ǜ,ǟǞǝ,ǟǞǝ,ǟ,ǢǡǠ,ǢǡǠ,Ǣ,ǥǤǣ,ǥǤǣ,ǥ,ǨǧǦ,ǨǧǦ,Ǩ,ǫǪǩ,ǫǪǩ,ǫ,ǮǭǬ,ǮǭǬ,Ǯ,Ǳǰǯ,Ǳǰǯ,Ǳ,Ǵǳǲ,Ǵǳǲ,Ǵ,ǷǶǵ,ǷǶǵ,Ƿ,ǺǹǸ,ǺǹǸ,Ǻ,ǽǼǻ,ǽǼǻ,ǽ,ȀǿǾ,ȀǿǾ,Ȁ,ȃȂȁ,ȃȂȁ,ȃ,ȆȅȄ,ȆȅȄ,Ȇ,ȉȈȇ,ȉȈȇ,ȉ,ȌȋȊ,ȌȋȊ,Ȍ,ȏȎȍ,ȏȎȍ,ȏ,ȒȑȐ,ȒȑȐ,Ȓ,ȕȔȓ,ȕȔȓ,ȕ,ȘȗȖ,ȘȗȖ,Ș,țȚș,țȚș,ț,ȞȝȜ,ȞȝȜ,Ȟ,ȡȠȟ,ȡȠȟ,ȡ,ȤȣȢ,ȤȣȢ,Ȥ,ȧȦȥ,ȧȦȥ,ȧ,ȯȨ,ȯȮ,Ȱȩ,Ȱȯ,ȱȪ,ȱȰ,Ȳȫ,Ȳȱ,ȳȬ,ȳȲ,ȴȮȨ,ȴȮȨ,ȵȩȴ,ȵȩȯ,ȵȴȯ,ȶȪȵ,ȶȪȰ,ȶȵȰ,ȷȫȶ,ȷȫȱ,ȷȶȱ,ȸȬȷ,ȸȬȲ,ȸȷȲ,ȹȭȸ,ȹȭȳ,ȹȸȳ,ȹ,ɁȺ,Ɂɀ,ɂȻ,ɂɁ,Ƀȼ,Ƀɂ,ɄȽ,ɄɃ,ɅȾ,ɅɄ,ɆɀȺ,ɆɀȺ,ɇȻɆ,ɇȻɁ,ɇɆɁ,Ɉȼɇ,Ɉȼɂ,Ɉɇɂ,ɉȽɈ,ɉȽɃ,ɉɈɃ,ɊȾɉ,ɊȾɄ,ɊɉɄ,ɋȿɊ,ɋȿɅ,ɋɊɅ,ɋ,ɏɌ,ɏɎ,ɐɎɌ,ɐɎɌ,ɑɍɐ,ɑɍɏ,ɑɐɏ,ɑ,ɕɒ,ɕɔ,ɖɔɒ,ɖɔɒ,ɗɓɖ,ɗɓɕ,ɗɖɕ,ɗ,ɚəɘ,ɚəɘ,ɚ,ɝɜɛ,ɝɜɛ,ɝ,ɠɟɞ,ɠɟɞ,ɠ,bÐ,bÜ,bë,bË,br,bV,bî,bo,bè,bÖ,bx,bi,bl,bY,bu,bñ,bÙ,bÓ,eǶ,eǭ,eǡ,eë,eȂ,eî,ei,eY,eÙ,eÓ,hÃ,hǆ,hǏ,hĤ,hĪ,ki,ni,qi,tG,tȚ,tȔ,ti,tļ,wĕ,wi,zi,Ai,Bi,Ci,Di,Ei,Fi,Fß,Ui,Xi,XĤ,XĪ,XƐ,Àĉ,ÀĤ,ÀĪ,Áĉ,ÁĤ,ÁĪ,Âĉ,ÂĤ,ÂĪ,Éë,Éƫ,Éƽ,ÉǕ,Éi,Éu,Éƍ,Éñ,Êë,Êƫ,Êƽ,ÊǕ,Êi,Êß,Êu,Êƍ,Êñ,Ïi,Ïȅ,ÒŚ,Òi,ÕĒ,Õi,ÕĤ,ÕĪ,Øi,Ûi,Þi,áŸ,ái,áÙ,âŸ,âi,âÙ,çi,çĤ,çĪ,êi,êÓ,êĤ,êĪ,íƊ,íi,íñ,ðG,ði,ðǿ,đĤ,đĪ,Ĕġ,Ĕİ,ĔĞ,Ĕĭ,ĔÃ,ĔĤ,ĔĪ,ĔĘ,Ĕħ,Ĕě,ėĤ,ėĪ,Ěĭ,ĚĤ,ĚĪ,Ěħ,ĝİ,ĝĤ,ĝĪ,ĝħ,ĠĤ,ģĪ,ĦĤ,ĦĪ,ĬĤ,ĬĪ,įĤ,įĪ,ĲĿ,Ĳë,ĲĶ,ĲĹ,Ĳi,Ĳl,ĲY,Ĳu,Ĳñ,ĲÙ,Ĳļ,ĵƓ,ĵǉ,ĵȈ,ĵi,ĵǌ,ĸi,ĸÙ,ĻƢ,ĻG,ĻŸ,ĻȔ,ĻǕ,ĻǛ,Ļi,ľi,ľñ,ŁŅ,Łŋ,Łő,Łi,Łß,ŁY,Łň,ŁŎ,ńi,Ňi,Ŋi,ŊŲ,ōȠ,ōȗ,ōr,ōi,ōÙ,Ői,Őñ,œũ,œŦ,œŚ,œŝ,œŬ,œů,œŠ,œi,œţ,ŖŔ,Ŗi,ři,Ŝi,şi,Ţi,ťi,Ũi,ūȋ,ūi,Ůi,űŵ,űi,űß,űĤ,űĪ,ŴȎ,Ŵi,ŴĤ,ŴĪ,ŷŵ,ŷÃ,ŷi,ŷĤ,ŷĪ,źƄ,źƁ,źž,źƇ,źi,Ži,ƀi,ƃi,ƃǼ,Ɔi,Ɖi,Ɖß,ƉĤ,ƉĪ,ƌi,ƌĤ,ƌĪ,ƌƐ,ƏĤ,ƏĪ,ƒŦ,ƒƙ,ƒƖ,ƒi,ƕi,ƕĤ,ƕĪ,Ƙi,ƘǼ,ƛi,ƛƟ,ƛĤ,ƛĪ,ƞi,ơŵ,ơi,ơĤ,ơĪ,Ƥƨ,ƤŔ,Ƥi,Ƥñ,Ƥň,Ƨi,ƧĤ,ƧĪ,ƪi,ƪƮ,ƪĤ,ƪĪ,ƭi,ƭĤ,ƭĪ,ưƷ,ưƴ,ưŔ,ưi,ƳŔ,Ƴi,ƶi,ƹi,ƹĤ,ƹĪ,Ƽi,ƼĤ,ƼĪ,ƿi,ƿu,ƿÙ,ƿļ,ǂǆ,ǂŔ,ǂi,ǂñ,ǅi,ǅñ,ǅȝ,ǈi,ǈƮ,ǈǌ,ǋi,ǎĤ,ǎĪ,ǑŔ,Ǒi,ǔŸ,ǔi,ǗŔ,Ǘi,ǚi,ǚǞ,ǝi,ǠŚ,Ǡi,ǣŔ,ǣi,ǦǪ,Ǧi,ǩi,Ǭǳ,Ǭǰ,Ǭi,ǯǳ,ǯi,ǲi,ǵǳ,ǵi,ǵñ,Ǹi,ǸÙ,ǻi,Ǿi,ȁi,Ȅi,ȇi,Ȋi,ȍi,Ȑi,ȓi,Ȗi,și,Ȝi,ȟi,ȨɎ,ȨȺ,ȨÃ,ȨĤ,ȨĪ,Ȩħ,ȩɎ,ȩȻ,ȩÃ,ȩĤ,ȩĪ,ȩħ,ȪɎ,Ȫȼ,ȪÃ,ȪĤ,ȪĪ,Ȫħ,ȫɍ,ȫȽ,ȫĞ,ȫɔ,ȫÃ,ȫĤ,ȫĪ,ȫħ,ȬɎ,ȬȾ,ȬĞ,Ȭɔ,ȬÃ,ȬĤ,ȬĪ,Ȭħ,ȭɎ,ȭȿ,ȭĞ,ȭɔ,ȭÃ,ȭĤ,ȭĪ,ȭħ,ȺĤ,ȺĪ,Ⱥħ,ɌĤ,ɌĪ,ɍÃ,ɍĤ,ɍĪ,ȻĤ,ȻĪ,Ȼħ,ȼɎ,ȼÃ,ȼĤ,ȼĪ,ȼħ,Ƚɍ,Ƚɔ,ȽÃ,ȽĤ,ȽĪ,ɒĤ,ɒĪ,ɒħ,ɓĤ,ɓĪ,ɓħ,ȾɎ,Ⱦɔ,ȾÃ,ȾĤ,ȾĪ,ȿɎ,ȿɔ,ȿÃ,ȿĤ,ȿĪ->'

    tensor_names = []
    tensors = {}
    for tnum, tensor in enumerate(arrays):
        tensor_names.append(f"E{tnum}")
        tensors[f"E{tnum}"] = tensor

    return format_string, tensor_names, tensors


def time_hyper_query(query, parameters):
    mode = "compiled" if parameters["initial_compilation_mode"] == "o" else "interpreted"

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters=parameters) as hyper:
        with Connection(hyper.endpoint, "data.hyper", CreateMode.CREATE_AND_REPLACE) as connection:
            # plan
            tic = timer()
            hyper_res = connection.execute_list_query("EXPLAIN " + query)
            toc = timer()

            time_planning = toc - tic

            # execute
            tic = timer()
            hyper_res = connection.execute_list_query(query)
            toc = timer()
    
    print(f"hyper ({mode}) result: {hyper_res}\n(computed in {toc - tic - time_planning}s) - planning time: {time_planning}")


def time_sqlite_query(query):
    db_connection = sql.connect("test.db")
    db = db_connection.cursor()
    res = db.execute(query)

    # plan
    tic = timer()
    sql_res = db.execute("EXPLAIN QUERY PLAN " + query)
    sql_res = sql_res.fetchall()
    toc = timer()

    time_planning = toc - tic

    # execute
    tic = timer()
    sql_res = db.execute(query)
    sql_res = sql_res.fetchall()
    toc = timer()
    
    print(f"sqlite result: {sql_res}\n(computed in {toc - tic - time_planning}s) - planning time: {time_planning}")


if __name__ == "__main__":
    QUERY_PATH = False
    SAT_952 = True
    sql_file = "test_query.sql"

    if QUERY_PATH:
        with open(sql_file, "r") as file:
            query = file.read()
    else:
        if SAT_952:
            einsum_notation, tensor_names, tensors = create_sat_952()
        else:
            einsum_notation = "ij,kj,k->i"

            tensor_names = ["A", "B", "v"]
            tensors = {
                "A": np.array([[0, 1, 0, 6], [19, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 4]]),
                "B": np.array([[0, 0, 5, 0], [0, 1, 0, 0], [0, 0, 18, 0], [0, 0, 0, 8]]),
                "v": np.array([1, 0, 9, 11])
            }
        
        query = sql_einsum_query_opt(einsum_notation, tensor_names, tensors)

        print(query)

    # ----- HYPER ------
        
    # hyper - compiled
    parameters = {
        "log_config": "",
        "max_query_size": "100000000",
        "hard_concurrent_query_thread_limit": "1",
        "initial_compilation_mode": "o"
    }

    time_hyper_query(query=query, parameters=parameters)

    # hyper - interpreted
    parameters = {
        "log_config": "",
        "max_query_size": "100000000",
        "hard_concurrent_query_thread_limit": "1",
        "initial_compilation_mode": "v"
    }

    time_hyper_query(query=query, parameters=parameters)


    # ------ SQLite ------

    time_sqlite_query(query)


    # ------ MonetDB ------

    # connection = pymonetdb.connect(username="monetdb", password="monetdb", hostname="localhost", database="demo")
    # cursor = connection.cursor()
    # cursor.execute(query)
    # res = cursor.fetchall()

    # print(res)