# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a_dafne_trainer = Analysis(['..\\dafne_trainer'],
             pathex=['..\\src'],
             binaries=[],
             datas=[('..\\LICENSE', '.'), ('..\\src\\dafne_models\\resources\\*', 'resources\\')],
			 hiddenimports = ['pydicom',
			    'dafne_models',
				'SimpleITK',
				'tensorflow',
				'skimage',
				'nibabel',
				'dafne_dl',
                'PyQt5',
                'matplotlib']
             hookspath=['..\\..\\dafne\\pyinstaller_hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz_dafne_trainer = PYZ(a_dafne_trainer.pure, a_dafne_trainer.zipped_data,
             cipher=block_cipher)
exe_dafne_trainer = EXE(pyz_dafne_trainer,
          a_dafne_trainer.scripts,
          [],
          exclude_binaries=True,
          name='dafne_trainer',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          icon='dafne_trainer_icon.ico',
          console=True)
coll_dafne_trainer = COLLECT(exe_dafne_trainer,
               a_dafne_trainer.binaries,
               a_dafne_trainer.zipfiles,
               a_dafne_trainer.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='dafne_trainer')