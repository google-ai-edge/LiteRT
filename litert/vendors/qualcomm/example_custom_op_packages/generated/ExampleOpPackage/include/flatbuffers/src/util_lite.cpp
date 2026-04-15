// Minimal flatbuffers util providing only ClassicLocale definition.
// The full util.cpp requires POSIX file I/O (mkdir, realpath) which is
// unavailable on Hexagon QuRT. This file provides just what flexbuffers needs.

#include <clocale>
#include "flatbuffers/util.h"

#if defined(FLATBUFFERS_LOCALE_INDEPENDENT) && \
    (FLATBUFFERS_LOCALE_INDEPENDENT > 0)

namespace flatbuffers {

ClassicLocale ClassicLocale::instance_;

ClassicLocale::ClassicLocale()
    : locale_(newlocale(LC_ALL, "C", nullptr)) {}
ClassicLocale::~ClassicLocale() { freelocale(locale_); }

}  // namespace flatbuffers

#endif  // !FLATBUFFERS_LOCALE_INDEPENDENT
