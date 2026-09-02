#!/usr/bin/env bash
# =====================================================================
# КОРПОРАТИВНЫЙ СКРИПТ ПОЛНОЙ ОЧИСТКИ СИСТЕМЫ (HARD RESET)
# Удаляет все неудачные артефакты установки Triton, Docker-слои и кэш.
# =====================================================================
set -e

echo "🚨 [WARNING] ВНИМАНИЕ: Вы запускаете тотальную очистку ИИ-инфраструктуры!"
echo "Это удалит все зависшие контейнеры, 19-гигабайтные образы Triton, скачанные архивы и освободит диск."
read -p "Вы уверены, что хотите продолжить? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ Ликвидация отменена пользователем."
    exit 0
fi

echo "🛑 [1/5] Принудительная остановка всех фоновых процессов и контейнеров..."
# Мягко тушим Triton, если он завис в фоне
pkill -9 -f tritonserver 2>/dev/null || true
pkill -9 -f perf_analyzer 2>/dev/null || true

# Находим и жестко удаляем любые контейнеры, связанные с tritonserver
FAILED_CONTAINERS=$(docker ps -a -q --filter "ancestor=tritonserver" --filter "ancestor=3154d419950c")
if [ -not -z "$FAILED_CONTAINERS" ]; then
    logger.info "   Удаление зависших контейнеров..."
    docker rm -f $FAILED_CONTAINERS 2>/dev/null || true
fi

echo "🗑️ [2/5] Удаление тяжелых Docker-образов (Освобождение ~20-40 ГБ)..."
# Удаляем кастомный образ, если он успел частично собраться
docker rmi -f tritonserver:25.01-custom 2>/dev/null || true
# Удаляем исходный 19-гигабайтный минимальный образ (IMAGE ID: 3154d419950c)
docker rmi -f 3154d419950c 2>/dev/null || true
# Глубокая очистка кэша сборщика Docker Buildx (удаляет зависшие слои apt-get)
docker builder prune -a -f

echo "🧹 [3/5] Глубокая очистка дискового пространства Docker (System Prune)..."
# Удаляет все неиспользуемые контейнеры, сети и кэш слоев сборки
docker system prune -a -f --volumes

echo "📂 [4/5] Удаление временных папок, распакованных дистрибутивов и архивов..."
# Удаляем остатки неудачных ручных распаковок по абсолютным путям проекта
rm -rf ./triton_dist/
rm -rf ./triton_clients/
rm -rf ./triton_oci_local/
rm -rf ./skopeo_cache/

# Удаляем скачанные тяжелые архивы с GitHub (.tar.gz и .zip)
rm -f v2.54.0-ubuntu2404.backends.tar.gz 2>/dev/null || true
rm -f v2.54.0-ubuntu2204.backends.tar.gz 2>/dev/null || true
rm -f v2.43.0-ubuntu2204.backends.tar.gz 2>/dev/null || true
rm -f tritonserver2.54.0-vllm.tar.gz 2>/dev/null || true
rm -f v2.54.0_ubuntu2404.clients.tar.gz 2>/dev/null || true
rm -f *.zip 2>/dev/null || true
rm -f index.html* 2>/dev/null || true

echo "🔒 [5/5] Очистка системных путей /usr/local от бинарников-инъекций..."
# Бережно вычищаем только те файлы, которые мы могли скопировать туда вручную
sudo rm -f /usr/local/bin/tritonserver 2>/dev/null || true
sudo rm -f /usr/local/bin/perf_analyzer 2>/dev/null || true
sudo rm -rf /usr/local/backends 2>/dev/null || true

# Синхронизируем состояние кэша системного линкера динамических библиотек
sudo ldconfig

print("\n" + "="*70)
echo "🏁 [SUCCESS] ХАРД-РЕЗЕТ И КЛИНИНГ НОУТБУКА УСПЕШНО ЗАВЕРШЕНЫ!"
echo "Все битые образы, скрытые слои сборщика и тяжелые архивы полностью удалены."
echo "Диск вашего ноутбука очищен и готов к дальнейшей работе."
print("="*70)
