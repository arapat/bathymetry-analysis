python analysis/get-fnr-fixed-fpr.py SIO all ./logs/test-all/runtime_scores/model_SIO_test_all_scores.pkl > model_SIO_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC all ./logs/test-all/runtime_scores/model_NGDC_test_all_scores.pkl > model_NGDC_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA all ./logs/test-all/runtime_scores/model_NGA_test_all_scores.pkl > model_NGA_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC all ./logs/test-all/runtime_scores/model_JAMSTEC_test_all_scores.pkl > model_JAMSTEC_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas all ./logs/test-all/runtime_scores/model_NOAA_geodas_test_all_scores.pkl > model_NOAA_geodas_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py all all ./logs/test-all/runtime_scores/model_all_test_all_scores.pkl > model_all_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi all ./logs/test-all/runtime_scores/model_US_multi_test_all_scores.pkl > model_US_multi_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO all ./logs/test-all/runtime_scores/model_AGSO_test_all_scores.pkl > model_AGSO_test_all_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC US_multi ./logs/cross-regions/runtime_scores/model_JAMSTEC_test_US_multi_scores.pkl > model_JAMSTEC_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA NOAA_geodas ./logs/cross-regions/runtime_scores/model_NGA_test_NOAA_geodas_scores.pkl > model_NGA_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py SIO AGSO ./logs/cross-regions/runtime_scores/model_SIO_test_AGSO_scores.pkl > model_SIO_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py SIO NOAA_geodas ./logs/cross-regions/runtime_scores/model_SIO_test_NOAA_geodas_scores.pkl > model_SIO_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi SIO ./logs/cross-regions/runtime_scores/model_US_multi_test_SIO_scores.pkl > model_US_multi_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas US_multi ./logs/cross-regions/runtime_scores/model_NOAA_geodas_test_US_multi_scores.pkl > model_NOAA_geodas_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas JAMSTEC ./logs/cross-regions/runtime_scores/model_NOAA_geodas_test_JAMSTEC_scores.pkl > model_NOAA_geodas_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO JAMSTEC ./logs/cross-regions/runtime_scores/model_AGSO_test_JAMSTEC_scores.pkl > model_AGSO_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas NGDC ./logs/cross-regions/runtime_scores/model_NOAA_geodas_test_NGDC_scores.pkl > model_NOAA_geodas_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC NGDC ./logs/cross-regions/runtime_scores/model_JAMSTEC_test_NGDC_scores.pkl > model_JAMSTEC_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas SIO ./logs/cross-regions/runtime_scores/model_NOAA_geodas_test_SIO_scores.pkl > model_NOAA_geodas_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC NGA ./logs/cross-regions/runtime_scores/model_JAMSTEC_test_NGA_scores.pkl > model_JAMSTEC_test_NGA_scores.txt &
python analysis/get-fnr-fixed-fpr.py SIO US_multi ./logs/cross-regions/runtime_scores/model_SIO_test_US_multi_scores.pkl > model_SIO_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA NGA ./logs/cross-regions/runtime_scores/model_NGA_test_NGA_scores.pkl > model_NGA_test_NGA_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA NGDC ./logs/cross-regions/runtime_scores/model_NGA_test_NGDC_scores.pkl > model_NGA_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi AGSO ./logs/cross-regions/runtime_scores/model_US_multi_test_AGSO_scores.pkl > model_US_multi_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO SIO ./logs/cross-regions/runtime_scores/model_AGSO_test_SIO_scores.pkl > model_AGSO_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA JAMSTEC ./logs/cross-regions/runtime_scores/model_NGA_test_JAMSTEC_scores.pkl > model_NGA_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi US_multi ./logs/cross-regions/runtime_scores/model_US_multi_test_US_multi_scores.pkl > model_US_multi_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC NGA ./logs/cross-regions/runtime_scores/model_NGDC_test_NGA_scores.pkl > model_NGDC_test_NGA_scores.txt &
python analysis/get-fnr-fixed-fpr.py SIO NGA ./logs/cross-regions/runtime_scores/model_SIO_test_NGA_scores.pkl > model_SIO_test_NGA_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC US_multi ./logs/cross-regions/runtime_scores/model_NGDC_test_US_multi_scores.pkl > model_NGDC_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO NGDC ./logs/cross-regions/runtime_scores/model_AGSO_test_NGDC_scores.pkl > model_AGSO_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi NOAA_geodas ./logs/cross-regions/runtime_scores/model_US_multi_test_NOAA_geodas_scores.pkl > model_US_multi_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC JAMSTEC ./logs/cross-regions/runtime_scores/model_NGDC_test_JAMSTEC_scores.pkl > model_NGDC_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA US_multi ./logs/cross-regions/runtime_scores/model_NGA_test_US_multi_scores.pkl > model_NGA_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC AGSO ./logs/cross-regions/runtime_scores/model_NGDC_test_AGSO_scores.pkl > model_NGDC_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py SIO NGDC ./logs/cross-regions/runtime_scores/model_SIO_test_NGDC_scores.pkl > model_SIO_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi JAMSTEC ./logs/cross-regions/runtime_scores/model_US_multi_test_JAMSTEC_scores.pkl > model_US_multi_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC JAMSTEC ./logs/cross-regions/runtime_scores/model_JAMSTEC_test_JAMSTEC_scores.pkl > model_JAMSTEC_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO NOAA_geodas ./logs/cross-regions/runtime_scores/model_AGSO_test_NOAA_geodas_scores.pkl > model_AGSO_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO NGA ./logs/cross-regions/runtime_scores/model_AGSO_test_NGA_scores.pkl > model_AGSO_test_NGA_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA AGSO ./logs/cross-regions/runtime_scores/model_NGA_test_AGSO_scores.pkl > model_NGA_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi NGDC ./logs/cross-regions/runtime_scores/model_US_multi_test_NGDC_scores.pkl > model_US_multi_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py SIO SIO ./logs/cross-regions/runtime_scores/model_SIO_test_SIO_scores.pkl > model_SIO_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC SIO ./logs/cross-regions/runtime_scores/model_NGDC_test_SIO_scores.pkl > model_NGDC_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas AGSO ./logs/cross-regions/runtime_scores/model_NOAA_geodas_test_AGSO_scores.pkl > model_NOAA_geodas_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC AGSO ./logs/cross-regions/runtime_scores/model_JAMSTEC_test_AGSO_scores.pkl > model_JAMSTEC_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas NOAA_geodas ./logs/cross-regions/runtime_scores/model_NOAA_geodas_test_NOAA_geodas_scores.pkl > model_NOAA_geodas_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO US_multi ./logs/cross-regions/runtime_scores/model_AGSO_test_US_multi_scores.pkl > model_AGSO_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGA SIO ./logs/cross-regions/runtime_scores/model_NGA_test_SIO_scores.pkl > model_NGA_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py NOAA_geodas NGA ./logs/cross-regions/runtime_scores/model_NOAA_geodas_test_NGA_scores.pkl > model_NOAA_geodas_test_NGA_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC SIO ./logs/cross-regions/runtime_scores/model_JAMSTEC_test_SIO_scores.pkl > model_JAMSTEC_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py AGSO AGSO ./logs/cross-regions/runtime_scores/model_AGSO_test_AGSO_scores.pkl > model_AGSO_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py JAMSTEC NOAA_geodas ./logs/cross-regions/runtime_scores/model_JAMSTEC_test_NOAA_geodas_scores.pkl > model_JAMSTEC_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py SIO JAMSTEC ./logs/cross-regions/runtime_scores/model_SIO_test_JAMSTEC_scores.pkl > model_SIO_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC NGDC ./logs/cross-regions/runtime_scores/model_NGDC_test_NGDC_scores.pkl > model_NGDC_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py NGDC NOAA_geodas ./logs/cross-regions/runtime_scores/model_NGDC_test_NOAA_geodas_scores.pkl > model_NGDC_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py US_multi NGA ./logs/cross-regions/runtime_scores/model_US_multi_test_NGA_scores.pkl > model_US_multi_test_NGA_scores.txt &
python analysis/get-fnr-fixed-fpr.py all SIO ./logs/cross-regions-all/runtime_scores/model_all_test_SIO_scores.pkl > model_all_test_SIO_scores.txt &
python analysis/get-fnr-fixed-fpr.py all US_multi ./logs/cross-regions-all/runtime_scores/model_all_test_US_multi_scores.pkl > model_all_test_US_multi_scores.txt &
python analysis/get-fnr-fixed-fpr.py all JAMSTEC ./logs/cross-regions-all/runtime_scores/model_all_test_JAMSTEC_scores.pkl > model_all_test_JAMSTEC_scores.txt &
python analysis/get-fnr-fixed-fpr.py all NGDC ./logs/cross-regions-all/runtime_scores/model_all_test_NGDC_scores.pkl > model_all_test_NGDC_scores.txt &
python analysis/get-fnr-fixed-fpr.py all NOAA_geodas ./logs/cross-regions-all/runtime_scores/model_all_test_NOAA_geodas_scores.pkl > model_all_test_NOAA_geodas_scores.txt &
python analysis/get-fnr-fixed-fpr.py all AGSO ./logs/cross-regions-all/runtime_scores/model_all_test_AGSO_scores.pkl > model_all_test_AGSO_scores.txt &
python analysis/get-fnr-fixed-fpr.py all NGA ./logs/cross-regions-all/runtime_scores/model_all_test_NGA_scores.pkl > model_all_test_NGA_scores.txt &

wait
