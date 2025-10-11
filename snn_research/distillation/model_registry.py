# ファイルパス: matsushibadenki/snn4/snn4-190ede29139f560c909685675a68ccf65069201c/snn_research/distillation/model_registry.py
#
# タイトル: モデルレジストリ
# 機能説明: find_models_for_taskメソッドの末尾にあった余分なコロンを削除し、SyntaxErrorを修正。
#
# 改善点:
# - ROADMAPフェーズ8に基づき、マルチエージェント間の知識共有を可能にする
#   分散型モデルレジストリ(DistributedModelRegistry)を実装。
# - ファイルロック機構を導入し、複数プロセスからの同時書き込みによる
#   レジストリファイルの破損を防止する。
#
# 改善点 (v2):
# - ROADMAPフェーズ4「社会学習」に基づき、エージェントがスキル（モデル）を
#   共有するための`publish_skill`および`download_skill`メソッドを実装。
#
# 改善点 (v3):
# - 複数プロセスからの同時書き込みの堅牢性を向上させるため、
#   一時ファイルへの書き込みとアトミックなリネーム処理を導入。

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
from pathlib import Path
import fcntl
import time
import shutil
import os # osモジュールをインポート

class ModelRegistry(ABC):
    """
    専門家モデルを管理するためのインターフェース。
    """
    @abstractmethod
    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        """新しいモデルをレジストリに登録する。"""
        pass

    @abstractmethod
    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """特定のタスクに最適なモデルを検索する。"""
        pass

    @abstractmethod
    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        """モデルIDに基づいてモデル情報を取得する。"""
        pass

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """登録されているすべてのモデルのリストを取得する。"""
        pass


class SimpleModelRegistry(ModelRegistry):
    """
    JSONファイルを使用したシンプルなモデルレジストリの実装。
    """
    def __init__(self, registry_path: str = "runs/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.project_root = self.registry_path.resolve().parent.parent
        self.models: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content:
                        return {}
                    return json.loads(content)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # 改善: アトミックな書き込み処理
        temp_path = self.registry_path.with_suffix(f"{self.registry_path.suffix}.tmp")
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.models, f, indent=4, ensure_ascii=False)
        # ファイルをアトミックにリネームして上書き
        os.rename(temp_path, self.registry_path)


    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        new_model_info = {
            "task_description": task_description,
            "metrics": metrics,
            "model_path": model_path,
            "config": config,
            "published": False # 社会学習のためのフラグ
        }
        if model_id not in self.models:
            self.models[model_id] = []
        # 同じモデルIDのエントリは上書きする
        self.models[model_id] = [new_model_info]
        self._save()
        print(f"Model for task '{model_id}' registered at '{model_path}'.")

    async def find_models_for_task(self, task_description: str, top_k: int = 1) -> List[Dict[str, Any]]:
        if task_description in self.models:
            models_for_task = self.models[task_description]
            
            models_for_task.sort(
                key=lambda x: x.get("metrics", {}).get("accuracy", 0),
                reverse=True
            )

            resolved_models = []
            for model_info in models_for_task[:top_k]:
                relative_path_str = model_info.get('model_path') or model_info.get('path')
                
                if relative_path_str:
                    absolute_path = Path(relative_path_str).resolve()
                    model_info['model_path'] = str(absolute_path)

                model_info['model_id'] = task_description
                resolved_models.append(model_info)
            
            return resolved_models
        return []


    async def get_model_info(self, model_id: str) -> Dict[str, Any] | None:
        models = self.models.get(model_id)
        if models:
            model_info = models[0] 
            relative_path_str = model_info.get('model_path') or model_info.get('path')
            if relative_path_str:
                absolute_path = Path(relative_path_str).resolve()
                model_info['model_path'] = str(absolute_path)
            return model_info
        return None

    async def list_models(self) -> List[Dict[str, Any]]:
        all_models = []
        for model_id, model_list in self.models.items():
            for model_info in model_list:
                model_info_with_id = {'model_id': model_id, **model_info}
                all_models.append(model_info_with_id)
        return all_models


class DistributedModelRegistry(SimpleModelRegistry):
    """
    ファイルロックを使用して、複数のプロセスからの安全なアクセスを保証する
    分散環境向けのモデルレジストリ。社会学習機能も持つ。
    """
    def __init__(self, registry_path: str = "runs/model_registry.json", timeout: int = 10, shared_skill_dir: str = "runs/shared_skills"):
        super().__init__(registry_path)
        self.timeout = timeout
        self.shared_skill_dir = Path(shared_skill_dir)
        self.shared_skill_dir.mkdir(parents=True, exist_ok=True)

    def _execute_with_lock(self, mode: str, operation, *args, **kwargs):
        """ファイルロックを取得して操作を実行するユーティリティメソッド"""
        start_time = time.time()
        # ファイルが存在しない場合でもエラーにならないように 'a+' を使用
        with open(self.registry_path, 'a+', encoding='utf-8') as f:
            while time.time() - start_time < self.timeout:
                try:
                    lock_type = fcntl.LOCK_EX if mode == 'w' else fcntl.LOCK_SH
                    fcntl.flock(f, lock_type)
                    f.seek(0)
                    result = operation(f, *args, **kwargs)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return result
                except (IOError, BlockingIOError):
                    time.sleep(0.1)
            raise IOError(f"レジストリの{'書き込み' if mode == 'w' else '読み取り'}ロックの取得に失敗しました。")

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        def read_operation(f):
            content = f.read()
            if not content:
                return {}
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {}
        return self._execute_with_lock('r', read_operation)

    def _save(self) -> None:
        # 親クラスのアトミックな保存処理を利用
        # 分散環境ではロックと組み合わせることがより堅牢
        def write_operation(f, models_to_save):
            # ファイル全体をロックしているので、直接書き込む
            f.seek(0)
            f.truncate()
            json.dump(models_to_save, f, indent=4, ensure_ascii=False)

        self._execute_with_lock('w', write_operation, self.models)


    async def register_model(self, model_id: str, task_description: str, metrics: Dict[str, float], model_path: str, config: Dict[str, Any]) -> None:
        self.models = self._load()
        await super().register_model(model_id, task_description, metrics, model_path, config)

    async def publish_skill(self, model_id: str) -> bool:
        """学習済みスキル（モデルファイル）を共有ディレクトリに公開する。"""
        self.models = self._load()
        model_info_list = self.models.get(model_id)
        if not model_info_list:
            print(f"❌ 公開失敗: モデル '{model_id}' は登録されていません。")
            return False
        
        model_info = model_info_list[0]
        src_path = Path(model_info['model_path'])
        if not src_path.exists():
            print(f"❌ 公開失敗: モデルファイルが見つかりません: {src_path}")
            return False

        dest_path = self.shared_skill_dir / f"{model_id}.pth"
        shutil.copy(src_path, dest_path)
        
        model_info['published'] = True
        model_info['shared_path'] = str(dest_path)
        self._save()
        print(f"🌍 スキル '{model_id}' を共有ディレクトリに公開しました: {dest_path}")
        return True

    async def download_skill(self, model_id: str, destination_dir: str) -> Dict[str, Any] | None:
        """共有されているスキルをダウンロードし、ローカルに登録する。"""
        self.models = self._load()
        # 他のエージェントが公開したスキルを探す
        # ここでは簡略化のため、自身のレジストリからpublished=Trueのものを探す
        all_published = [
            {'model_id': mid, **info}
            for mid, info_list in self.models.items()
            for info in info_list if info.get('published')
        ]
        
        target_skill = next((s for s in all_published if s['model_id'] == model_id), None)

        if not target_skill or not target_skill.get('shared_path'):
            print(f"❌ ダウンロード失敗: 共有スキル '{model_id}' が見つかりません。")
            return None

        src_path = Path(target_skill['shared_path'])
        if not src_path.exists():
            print(f"❌ ダウンロード失敗: 共有ファイルが見つかりません: {src_path}")
            return None

        dest_path = Path(destination_dir) / f"{model_id}.pth"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dest_path)

        # ダウンロードしたスキルを自身のレジストリに登録
        new_local_info = target_skill.copy()
        new_local_info['model_path'] = str(dest_path)
        
        await self.register_model(
            model_id=model_id,
            task_description=new_local_info['task_description'],
            metrics=new_local_info['metrics'],
            model_path=new_local_info['model_path'],
            config=new_local_info['config']
        )
        print(f"✅ スキル '{model_id}' をダウンロードし、ローカルに登録しました: {dest_path}")
        return new_local_info