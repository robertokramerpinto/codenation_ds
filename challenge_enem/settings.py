INITIAL_FEATURES = ['NU_INSCRICAO','SG_UF_RESIDENCIA', 'NU_IDADE',
       'TP_SEXO', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO',
       'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO',
       'TP_DEPENDENCIA_ADM_ESC', 'IN_BAIXA_VISAO', 'IN_CEGUEIRA', 'IN_SURDEZ',
       'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_SABATISTA', 'IN_GESTANTE',
       'IN_IDOSO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
       'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
       'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA',
       'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
       'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO', 'Q001', 'Q002',
       'Q006', 'Q024', 'Q025', 'Q026', 'Q027', 'Q047']

TARGET = 'NU_NOTA_MT'

CATEGORICAL_COLS = ['SG_UF_RESIDENCIA','TP_SEXO','TP_COR_RACA', 'TP_NACIONALIDADE',
                   'TP_ESCOLA','TP_ENSINO', 'IN_TREINEIRO','TP_DEPENDENCIA_ADM_ESC',
                   'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
                   'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
                    'TP_STATUS_REDACAO','Q001', 'Q002',
                   'Q006', 'Q024', 'Q025', 'Q026', 'Q027', 'Q047']

NUMERICAL_COLS = ['NU_IDADE','TP_ST_CONCLUSAO','TP_ANO_CONCLUIU','IN_BAIXA_VISAO',
                 'IN_CEGUEIRA', 'IN_SURDEZ','IN_DISLEXIA', 'IN_DISCALCULIA', 
                  'IN_SABATISTA', 'IN_GESTANTE','IN_IDOSO',
                 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC','TP_LINGUA',
                  'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
                   'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']

SELECTED_FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS